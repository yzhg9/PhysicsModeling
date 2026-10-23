from phys_utils.prompt_loader import load_prompt
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm.asyncio import tqdm_asyncio
from phys_utils.io import JsonlWriter, load_records, load_existing_attempts
from phys_utils.llm_client import AsyncLLMClient
MODELING_PROMPT_TEMPLATE = load_prompt('scripts.generate_modeling.MODELING_PROMPT_TEMPLATE')

def normalize_image_paths(rec: Dict[str, Any], mm_root: Optional[str]) -> Tuple[List[str], List[str]]:
    image_paths = rec.get('image_path')
    if not isinstance(image_paths, list):
        return ([], [])
    root = os.path.realpath(mm_root) if mm_root else None
    raw: List[str] = []
    resolved: List[str] = []
    for p in image_paths:
        if not isinstance(p, str) or not p:
            continue
        p = p.replace('\\', '/')
        raw.append(p)
        if p.startswith(('http://', 'https://', 'data:', 'file://')):
            resolved.append(p)
            continue
        if os.path.isabs(p):
            resolved.append(p)
            continue
        if root:
            resolved.append(os.path.join(root, p.lstrip('/')))
        else:
            raise ValueError(f'Relative image_path without mm_root: {p}')
    return (raw, resolved)

def safe_str(x: Any) -> str:
    return '' if x is None else str(x)

def safe_int(x: Any, default: int=1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def load_done_keys_resume(output_path: str) -> Set[Tuple[str, int]]:
    done_keys: Set[Tuple[str, int]] = set()
    if not output_path or not os.path.exists(output_path):
        return done_keys
    try:
        done_keys = load_existing_attempts(output_path, id_key='id', attempt_key='attempt')
    except Exception:
        done_keys = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = obj.get('id')
                if rid is None:
                    continue
                if 'attempt' not in obj or obj.get('attempt') is None:
                    done_keys.add((safe_str(rid), 1))
    except Exception:
        pass
    return done_keys

def load_type_info(type_path: Optional[str]) -> Dict[str, Dict[str, Any]]:

    if not type_path or not os.path.exists(type_path):
        return {}
    info: Dict[str, Dict[str, Any]] = {}
    try:
        objs = load_records(type_path)
    except Exception:
        objs = None

    def _put(rid_any: Any, typ: Any=None, norm: Any=None):
        rid = safe_str(rid_any)
        if not rid:
            return
        d = info.get(rid, {})
        if isinstance(typ, str) and typ.strip():
            d['type'] = typ.strip()
        if isinstance(norm, str) and norm.strip():
            d['normalized_answer'] = norm.strip()
        info[rid] = d
    if isinstance(objs, list):
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            _put(obj.get('id'), obj.get('type'), obj.get('normalized_answer'))
        if info:
            return info
    try:
        with open(type_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    _put(k, v, None)
                elif isinstance(v, dict):
                    _put(k, v.get('type'), v.get('normalized_answer'))
        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                _put(obj.get('id'), obj.get('type'), obj.get('normalized_answer'))
    except Exception:
        pass
    return info

def get_q_type(rec: Dict[str, Any], type_info: Dict[str, Dict[str, Any]]) -> str:

    rid = safe_str(rec.get('id'))
    ext = type_info.get(rid) or {}
    t = ext.get('type')
    if isinstance(t, str) and t.strip():
        return t.strip()
    v = rec.get('type')
    if isinstance(v, str) and v.strip():
        return v.strip()
    return 'Textual'

def get_typefile_normalized_answer(rec: Dict[str, Any], type_info: Dict[str, Dict[str, Any]]) -> Optional[str]:
    rid = safe_str(rec.get('id'))
    ext = type_info.get(rid) or {}
    na = ext.get('normalized_answer')
    if isinstance(na, str) and na.strip():
        return na.strip()
    return None

async def infer_modeling_for_record(modeling_client: AsyncLLMClient, rec: Dict[str, Any], type_info: Dict[str, Dict[str, Any]], mm_root: Optional[str], modeling_prompt_template: str) -> Dict[str, Any]:
    rid = safe_str(rec.get('id'))
    attempt_i = safe_int(rec.get('attempt'), 1)
    num_attempts = safe_int(rec.get('num_attempts'), 1)
    question = (rec.get('question') or '').strip()
    raw_gt_answer = (rec.get('answer') or rec.get('gt_answer') or '').strip()
    q_type = get_q_type(rec, type_info)
    typefile_norm = get_typefile_normalized_answer(rec, type_info)
    image_paths_raw, image_paths_resolved = normalize_image_paths(rec, mm_root)
    user_prompt = modeling_prompt_template.format(question=question)
    content, meta = await modeling_client.chat_generate(user_prompt=user_prompt, image_paths=image_paths_resolved if image_paths_resolved else None)
    if isinstance(meta.get('usage'), dict):
        meta['usage'].pop('completion_tokens_details', None)
        meta['usage'].pop('prompt_tokens_details', None)
    patch: Dict[str, Any] = {'id': rid, 'attempt': attempt_i, 'num_attempts': num_attempts, 'subject': rec.get('subject'), 'source': rec.get('source'), 'image_path': image_paths_raw, 'question': question, 'gt_answer': raw_gt_answer, 'normalized_answer': typefile_norm, 'type': q_type, 'modeling': content, 'response': content, 'modeling_meta': meta}
    out = dict(rec)
    out.update(patch)
    return out

async def run(input_path: str, output_path: str, model_name: str, temperature: float, top_p: float, max_concurrent: int, resume: bool, api_base: str, api_key: str, max_tokens: int, mm_root: Optional[str], type_path: Optional[str], modeling_provider: str, modeling_image_mode: str, modeling_prompt_path: Optional[str], num_attempts: int) -> None:
    modeling_prompt_template = MODELING_PROMPT_TEMPLATE
    if modeling_prompt_path:
        with open(modeling_prompt_path, 'r', encoding='utf-8') as f:
            modeling_prompt_template = f.read()
        if '{question}' not in modeling_prompt_template:
            raise ValueError("Custom modeling prompt must contain '{question}' placeholder.")
    modeling_client = AsyncLLMClient(model_name=model_name, provider=modeling_provider, base_url=api_base, api_key=api_key, temperature=temperature, top_p=top_p, max_tokens=max_tokens, max_concurrent_requests=max_concurrent, mm_root=mm_root, image_mode=modeling_image_mode)
    type_info = load_type_info(type_path)
    records: List[Dict[str, Any]] = load_records(input_path)
    done_keys: Set[Tuple[str, int]] = set()
    if resume:
        done_keys = load_done_keys_resume(output_path)
    todo: List[Dict[str, Any]] = []
    default_K = max(1, int(num_attempts))
    for rec in records:
        rid = safe_str(rec.get('id'))
        if not rid:
            continue
        if rec.get('attempt') is not None:
            att_i = safe_int(rec.get('attempt'), 1)
            if (rid, att_i) in done_keys:
                continue
            r2 = dict(rec)
            r2['attempt'] = att_i
            r2['num_attempts'] = safe_int(rec.get('num_attempts'), default_K)
            todo.append(r2)
            continue
        K = safe_int(rec.get('num_attempts'), default_K)
        for att_i in range(1, K + 1):
            if (rid, att_i) in done_keys:
                continue
            r2 = dict(rec)
            r2['attempt'] = att_i
            r2['num_attempts'] = K
            todo.append(r2)

    async def process(rec: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await infer_modeling_for_record(modeling_client=modeling_client, rec=rec, type_info=type_info, mm_root=mm_root, modeling_prompt_template=modeling_prompt_template)
        except Exception as e:
            rid = safe_str(rec.get('id'))
            att_i = safe_int(rec.get('attempt'), 1)
            out = dict(rec)
            out.update({'id': rid, 'attempt': att_i, 'modeling': None, 'response': None, 'modeling_meta': None, 'error': f'Physical modeling generation failed: {type(e).__name__}'})
            return out
    mode = 'a' if resume and os.path.exists(output_path) else 'w'
    with JsonlWriter(output_path, mode=mode, ensure_ascii=False) as writer:
        tasks = [asyncio.create_task(process(r)) for r in todo]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc='Generating modeling (id,attempt)'):
            result = await fut
            rid = safe_str(result.get('id'))
            att_i = safe_int(result.get('attempt'), 1)
            done_keys.add((rid, att_i))
            writer.write(result)

def parse_args():
    p = argparse.ArgumentParser(description='Generate physical modeling with AsyncLLMClient (no judging).')
    p.add_argument('--input', type=str, required=True, help='Path to input JSON/JSONL.')
    p.add_argument('--output', type=str, required=True, help='Path to output JSONL.')
    p.add_argument('--model', type=str, required=True, help='Physical modeling model name (vLLM served name or API model).')
    p.add_argument('--api_base', type=str, default=None)
    p.add_argument('--api_key', type=str, default=None)
    p.add_argument('--modeling_provider', type=str, default='auto', help='auto|vllm|openai|gemini|compat')
    p.add_argument('--modeling_image_mode', type=str, default='auto', help='auto|file_url|data_url')
    p.add_argument('--mm_root', type=str, default='data', help='Dataset/media root used to resolve relative image_path and to restrict local file reads.')
    p.add_argument('--max_tokens', type=int, default=10240)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--max_concurrent', type=int, default=16)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--num_attempts', type=int, default=1, help="Generate K attempts per id if input doesn't specify.")
    p.add_argument('--type_file', type=str, default=None, help='Optional JSON/JSONL file providing {"id": ..., "type": ...}. If set, overrides record["type"].')
    p.add_argument('--modeling_prompt', type=str, default=None, help='Optional path to a plain-text prompt template. Must include {question}.')
    return p.parse_args()

def main():
    args = parse_args()
    asyncio.run(run(input_path=args.input, output_path=args.output, model_name=args.model, temperature=args.temperature, top_p=args.top_p, max_concurrent=args.max_concurrent, resume=args.resume, api_base=args.api_base, api_key=args.api_key, max_tokens=args.max_tokens, mm_root=args.mm_root, type_path=args.type_file, modeling_provider=args.modeling_provider, modeling_image_mode=args.modeling_image_mode, modeling_prompt_path=args.modeling_prompt, num_attempts=args.num_attempts))
if __name__ == '__main__':
    main()
