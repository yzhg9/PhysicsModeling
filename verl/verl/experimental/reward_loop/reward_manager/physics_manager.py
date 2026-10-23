# Modified for PhysReasoning: modeling rollout, physics rewards, metrics, or image handling.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Optional
import inspect
from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from phys_utils import AsyncLLMClient, EXECUTE_MODELING_PROMPT

@register('physics_manager')
class PhysicsRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None, remote_model_name='qwen3.5-flash', backup_model_name='qwen3.5-flash', **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.remote_model = AsyncLLMClient(model_name=remote_model_name, max_concurrent_requests=16)
        self.backup_model = AsyncLLMClient(model_name=backup_model_name, max_concurrent_requests=16)
        reward_cfg = getattr(self.config.reward, 'reward_kwargs', None)
        self.modeling_execute = _get_with_fallback(kwargs, reward_cfg, 'modeling_execute', False)
        self.remote_execute = _get_with_fallback(kwargs, reward_cfg, 'remote_execute', False)
        if self.modeling_execute and self.remote_execute:
            remote_executor_name = _get_with_fallback(kwargs, reward_cfg, 'remote_executor_name', remote_model_name)
            self.remote_executor = AsyncLLMClient(model_name=remote_executor_name)
        else:
            self.remote_executor = None
        self.enable_rubric = _get_with_fallback(kwargs, reward_cfg, 'enable_rubric', True)
        self.enable_correctness = _get_with_fallback(kwargs, reward_cfg, 'enable_correctness', True)
        self.normalize_rubric = _get_with_fallback(kwargs, reward_cfg, 'normalize_rubric', True)
        self.rubric_weight = _get_with_fallback(kwargs, reward_cfg, 'rubric_weight', 1.0)
        self.correctness_weight = _get_with_fallback(kwargs, reward_cfg, 'correctness_weight', 1.0)
        self.format_weight = _get_with_fallback(kwargs, reward_cfg, 'format_weight', 0.5)
        self.enable_reflection = _get_with_fallback(kwargs, reward_cfg, 'enable_reflection', False)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, 'Only support single data item'
        data_item = data[0]
        response_ids = data_item.batch['responses']
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch['attention_mask'][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        data_source = data_item.non_tensor_batch['data_source']
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        tool_extra_fields = data_item.non_tensor_batch.get('tool_extra_fields', None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
        num_turns = data_item.non_tensor_batch.get('__num_turns__', None)
        rollout_reward_scores = data_item.non_tensor_batch.get('reward_scores', {})
        extra_info['num_turns'] = num_turns
        extra_info['rollout_reward_scores'] = rollout_reward_scores
        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))
        if self.modeling_execute and self.remote_execute:
            question = extra_info['question']
            modeling_part = _extract_modeling_block(response_str)
            modeling = modeling_part if modeling_part else response_str
            remote_execution, remote_execution_meta = await self.remote_executor.chat_generate(user_prompt=EXECUTE_MODELING_PROMPT.format(question=question, modeling=modeling))
            extra_info['remote_execution'] = remote_execution
        extra_reward_kwargs = {'rubric_weight': self.rubric_weight, 'correctness_weight': self.correctness_weight, 'format_weight': self.format_weight}
        if self.reward_router_address is not None:
            extra_reward_kwargs['reward_router_address'] = self.reward_router_address
            extra_reward_kwargs['reward_model_tokenizer'] = self.reward_model_tokenizer
        result = await self.compute_score(response=response_str, ground_truth=ground_truth, extra_info=extra_info, model=self.remote_model, backup_model=self.backup_model, enable_rubric=self.enable_rubric, enable_correctness=self.enable_correctness, normalize_rubric=self.normalize_rubric, modeling_execute=self.modeling_execute, remote_execute=self.remote_execute, enable_reflection=self.enable_reflection, **extra_reward_kwargs)
        reward_extra_info = {}
        score: float
        if isinstance(result, dict):
            score = result['score']
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info['acc'] = score
        reward = score
        return {'reward_score': reward, 'reward_extra_info': reward_extra_info}

def _extract_tag_block(text: str, tag: str) -> Optional[str]:
    """
    Extract inner content of a single XML-like block:
    <tag> ... </tag>

    Returns:
        - stripped inner content if found and non-empty
        - None if tag not found or content is empty
    """
    s = (text or '').strip()
    if not s:
        return None
    pattern = f'<{tag}>(.*?)</{tag}>'
    m = re.search(pattern, s, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None

def _extract_modeling_block(text: str) -> Optional[str]:
    return _extract_tag_block(text, 'modeling')

def _get_with_fallback(kwargs, config_obj, key, default):
    if key in kwargs:
        return kwargs[key]
    if config_obj is not None:
        value = getattr(config_obj, key, None)
        if value is not None:
            return value
    return default
