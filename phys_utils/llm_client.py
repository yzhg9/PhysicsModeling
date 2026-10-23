from __future__ import annotations
import argparse
import asyncio
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
GEMINI_API_KEY_ENV = 'GEMINI_API_KEY'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
DASHSCOPE_API_KEY_ENV = 'DASHSCOPE_API_KEY'
GEMINI_BASE_URL_DEFAULT = 'https://generativelanguage.googleapis.com/v1beta/openai/'
OPENAI_BASE_URL_DEFAULT = 'https://api.openai.com/v1/'
QWEN_BASE_URL_DEFAULT = ''
default_retry = retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=32), retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)), reraise=True)

@dataclass
class LLMResult:
    content: str
    meta: Dict[str, Any]

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    if mt:
        return mt
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    if ext == '.webp':
        return 'image/webp'
    return 'image/png'

def _load_image_as_data_url(path: str, max_side: int=448, jpeg_quality: int=85) -> str:
    """
    Encode local image as base64 data URL (JPEG).
    Good for remote APIs. Larger request bodies than file://.
    """
    img = Image.open(path).convert('RGB')
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'

def _looks_like_local_vllm(base_url: str) -> bool:
    u = base_url.lower()
    return any((x in u for x in ['127.0.0.1', 'localhost', '0.0.0.0', '[::1]']))

class AsyncLLMClient:
    """
    One unified OpenAI-compatible async chat client.

    - Remote APIs: images -> base64 data URL
    - Local vLLM:  images -> file://abs_path  (requires vLLM allowed-local-media-path on server side)

    Output is unified: (text, meta)
    """

    def __init__(self, model_name: str, provider: str='auto', api_key: Optional[str]=None, base_url: Optional[str]=None, temperature: float=0.0, top_p: float=1.0, max_tokens: Optional[int]=None, max_concurrent_requests: int=5, system_prompt: Optional[str]=None, mm_root: Optional[str]=None, image_mode: str='auto', data_url_max_side: int=448, data_url_jpeg_quality: int=85):
        self.model_name = model_name
        self.provider = provider.lower()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.mm_root = mm_root
        self.image_mode = image_mode.lower()
        self.data_url_max_side = data_url_max_side
        self.data_url_jpeg_quality = data_url_jpeg_quality
        final_base_url, final_api_key = self._resolve_credentials(model_name=model_name, provider=self.provider, base_url=base_url, api_key=api_key)
        self.base_url = final_base_url
        self.client = AsyncOpenAI(base_url=final_base_url, api_key=final_api_key)
        self._loop = None
        self._semaphore = None
        self.max_concurrent_requests = max_concurrent_requests

    def _resolve_credentials(self, model_name: str, provider: str, base_url: Optional[str], api_key: Optional[str]) -> Tuple[str, str]:
        if base_url:
            final_base_url = base_url
        else:
            m = model_name.lower()
            if provider in ('gemini',) or m.startswith('gemini'):
                final_base_url = os.getenv('GEMINI_BASE_URL', GEMINI_BASE_URL_DEFAULT)
            elif provider in ('qwen', 'dashscope') or m.startswith('qwen'):
                final_base_url = os.getenv('DASHSCOPE_BASE_URL', QWEN_BASE_URL_DEFAULT)
            elif provider in ('openai',) or m.startswith(('gpt', 'o1', 'o3')):
                final_base_url = os.getenv('OPENAI_BASE_URL', OPENAI_BASE_URL_DEFAULT)
            elif provider in ('vllm',):
                final_base_url = 'http://127.0.0.1:8000/v1'
            else:
                final_base_url = os.getenv('OPENAI_BASE_URL', OPENAI_BASE_URL_DEFAULT)
        if api_key:
            final_api_key = api_key
        elif provider == 'vllm' or _looks_like_local_vllm(final_base_url):
            final_api_key = 'EMPTY'
        else:
            m = model_name.lower()
            if provider == 'gemini' or m.startswith('gemini'):
                final_api_key = os.getenv(GEMINI_API_KEY_ENV, '')
                if not final_api_key:
                    raise EnvironmentError(f'Missing {GEMINI_API_KEY_ENV}.')
            elif provider in ('qwen', 'dashscope') or m.startswith('qwen'):
                final_api_key = os.getenv(DASHSCOPE_API_KEY_ENV, '')
                if not final_api_key:
                    raise EnvironmentError(f'Missing {DASHSCOPE_API_KEY_ENV}.')
            else:
                final_api_key = os.getenv(OPENAI_API_KEY_ENV, '')
                if not final_api_key:
                    raise EnvironmentError(f'Missing {OPENAI_API_KEY_ENV}.')
        if not final_base_url:
            raise EnvironmentError('Set DASHSCOPE_BASE_URL for the selected provider.')
        return (final_base_url, final_api_key)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self._loop = loop
        return self._semaphore

    def _validate_media_path(self, path: str) -> str:
        if not path:
            return path
        if path.startswith(('http://', 'https://', 'data:', 'file://')):
            return path
        ap = os.path.abspath(path)
        if self.mm_root:
            real_ap = os.path.realpath(ap)
            real_root = os.path.realpath(self.mm_root)
            if not real_ap.startswith(real_root):
                candidate = os.path.realpath(os.path.join(real_root, path.lstrip('/')))
                if os.path.exists(candidate) and candidate.startswith(real_root):
                    ap = candidate
                else:
                    raise ValueError(f'Media path {ap} is outside allowed root {real_root}. Move files under root or set mm_root correctly.')
        return ap

    def _to_file_url(self, abs_path: str) -> str:
        return 'file://' + abs_path

    def _decide_image_mode(self) -> str:
        if self.image_mode in ('data_url', 'file_url'):
            return self.image_mode
        if self.provider == 'vllm' or _looks_like_local_vllm(self.base_url):
            return 'file_url'
        return 'data_url'

    def _build_user_message(self, user_prompt: str, image_paths: Optional[List[str]]) -> Dict[str, Any]:
        if not image_paths:
            return {'role': 'user', 'content': user_prompt}
        mode = self._decide_image_mode()
        content_payload: List[Dict[str, Any]] = [{'type': 'text', 'text': user_prompt}]
        for p in image_paths:
            if not p:
                continue
            safe_p = self._validate_media_path(p)
            if safe_p.startswith(('http://', 'https://', 'data:', 'file://')):
                url_val = safe_p
            else:
                if not os.path.exists(safe_p):
                    raise FileNotFoundError(f'Image not found: {safe_p}')
                if mode == 'file_url':
                    url_val = self._to_file_url(safe_p)
                else:
                    url_val = _load_image_as_data_url(safe_p, max_side=self.data_url_max_side, jpeg_quality=self.data_url_jpeg_quality)
            content_payload.append({'type': 'image_url', 'image_url': {'url': url_val}})
        return {'role': 'user', 'content': content_payload}

    @default_retry
    async def chat_generate(self, user_prompt: str, image_paths: Optional[List[str]]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, max_tokens: Optional[int]=None, extra_create_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[str, Dict[str, Any]]:
        """
        Unified output: (text, meta)
        meta includes finish_reason, usage, model; provider endpoints are omitted.
        """
        temp = self.temperature if temperature is None else temperature
        tp = self.top_p if top_p is None else top_p
        mt = self.max_tokens if max_tokens is None else max_tokens
        messages: List[Dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(self._build_user_message(user_prompt, image_paths))
        params: Dict[str, Any] = {'model': self.model_name, 'messages': messages, 'temperature': temp, 'top_p': tp}
        if mt is not None:
            params['max_tokens'] = mt
        if extra_create_kwargs:
            params.update(extra_create_kwargs)
        async with self.semaphore:
            resp = await self.client.chat.completions.create(**params)
        text = (resp.choices[0].message.content or '').strip()
        usage = getattr(resp, 'usage', None)
        if usage is None:
            usage_dict = None
        else:
            try:
                usage_dict = dict(usage)
            except Exception:
                usage_dict = json.loads(json.dumps(usage, default=str))
        meta = {'finish_reason': getattr(resp.choices[0], 'finish_reason', None), 'usage': usage_dict, 'model': self.model_name}
        return (text, meta)

    @default_retry
    async def chat_generate_thinking(self, user_prompt: str, image_paths: Optional[List[str]]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, max_tokens: Optional[int]=None, extra_create_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[str, Dict[str, Any]]:
        """
        Unified output: (text, meta)
        meta includes finish_reason, usage, model; provider endpoints are omitted.
        """
        temp = self.temperature if temperature is None else temperature
        tp = self.top_p if top_p is None else top_p
        mt = self.max_tokens if max_tokens is None else max_tokens
        messages: List[Dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(self._build_user_message(user_prompt, image_paths))
        params: Dict[str, Any] = {'model': self.model_name, 'messages': messages, 'temperature': temp, 'top_p': tp}
        if mt is not None:
            params['max_tokens'] = mt
        if self.provider in ('qwen', 'dashscope') or self.model_name.lower().startswith('qwen'):
            params.setdefault('extra_body', {})['enable_thinking'] = True
        if extra_create_kwargs:
            params.update(extra_create_kwargs)
        async with self.semaphore:
            resp = await self.client.chat.completions.create(**params)
            msg = resp.choices[0].message
            text = (msg.content or '').strip()
            thinking_text = (getattr(msg, 'reasoning_content', None) or '').strip()
        usage = getattr(resp, 'usage', None)
        if usage is None:
            usage_dict = None
        else:
            try:
                usage_dict = dict(usage)
            except Exception:
                usage_dict = json.loads(json.dumps(usage, default=str))
        meta = {'finish_reason': getattr(resp.choices[0], 'finish_reason', None), 'usage': usage_dict, 'model': self.model_name}
        return (thinking_text, text, meta)

    @default_retry
    async def chat_generate_messages(self, messages: List[Dict[str, Any]], temperature: Optional[float]=None, top_p: Optional[float]=None, max_tokens: Optional[int]=None, extra_create_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[str, Dict[str, Any]]:
        """
        Expect messages already in OpenAI chat format.
        If the first message isn't system and self.system_prompt is set, we prepend it.
        """
        temp = self.temperature if temperature is None else temperature
        tp = self.top_p if top_p is None else top_p
        mt = self.max_tokens if max_tokens is None else max_tokens
        if self.system_prompt is not None:
            if not messages or messages[0].get('role') != 'system':
                messages = [{'role': 'system', 'content': self.system_prompt}] + messages
        params: Dict[str, Any] = {'model': self.model_name, 'messages': messages, 'temperature': temp, 'top_p': tp}
        if mt is not None:
            params['max_tokens'] = mt
        if extra_create_kwargs:
            params.update(extra_create_kwargs)
        async with self.semaphore:
            resp = await self.client.chat.completions.create(**params)
        text = (resp.choices[0].message.content or '').strip()
        usage = getattr(resp, 'usage', None)
        if usage is None:
            usage_dict = None
        else:
            try:
                usage_dict = dict(usage)
            except Exception:
                usage_dict = json.loads(json.dumps(usage, default=str))
        meta = {'finish_reason': getattr(resp.choices[0], 'finish_reason', None), 'usage': usage_dict, 'model': self.model_name}
        return (text, meta)

    async def batch_chat_generate(self, batch_data: List[Union[str, Tuple[str, Optional[List[str]]]]], temperature: Optional[float]=None, top_p: Optional[float]=None, max_tokens: Optional[int]=None, return_exceptions: bool=True) -> List[Tuple[str, Dict[str, Any]]]:

        async def one(item):
            try:
                if isinstance(item, str):
                    return await self.chat_generate(item, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                if isinstance(item, tuple):
                    prompt, images = item
                    return await self.chat_generate(prompt, image_paths=images, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                raise ValueError('Invalid batch item format.')
            except Exception as e:
                if return_exceptions:
                    return (f'[Error: {type(e).__name__}]', {'finish_reason': 'error', 'usage': None, 'error': type(e).__name__})
                raise
        return await asyncio.gather(*(one(x) for x in batch_data))
AsyncGeminiClient = AsyncLLMClient
AsyncVLLMModel = AsyncLLMClient
