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

import logging
import os
import re
from typing import Any
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_LOGGING_LEVEL', 'WARN'))

@register('modeling_only_agent')
class ModelingOnlyAgentLoop(AgentLoopBase):
    """Single-turn rollout with a response mask over physical modeling tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        print(f'[COMPUTE MODELING LOGP ONLY] Activated!!!')

    def _find_modeling_char_spans(self, text: str) -> list[tuple[int, int]]:
        """Find all <modeling>...</modeling> spans in decoded text, inclusive of the tags."""
        spans = []
        for match in re.finditer('<modeling>.*?</modeling>', text, flags=re.DOTALL):
            spans.append((match.start(), match.end()))
        return spans

    def _build_modeling_mask_from_offsets(self, response_ids: list[int], decoded_text: str) -> list[int] | None:
        modeling_spans = self._find_modeling_char_spans(decoded_text)
        if not modeling_spans:
            return [0] * len(response_ids)
        try:
            encoded = self.tokenizer(decoded_text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception as e:
            logger.warning(f'Failed to build modeling mask from offsets: {type(e).__name__}')
            return None
        encoded_ids = encoded.get('input_ids')
        offsets = encoded.get('offset_mapping')
        if encoded_ids is None or offsets is None:
            return None
        aligned_offsets = None
        if encoded_ids == response_ids:
            aligned_offsets = offsets
        elif len(response_ids) == len(encoded_ids) + 1 and response_ids[:len(encoded_ids)] == encoded_ids:
            aligned_offsets = list(offsets) + [(len(decoded_text), len(decoded_text))]
        else:
            logger.warning(f'Retokenized decoded response is not safely aligned with generated response. Falling back to prefix-based alignment. retokenized={len(encoded_ids)}, generated={len(response_ids)}')
            return None
        mask = [0] * len(response_ids)
        for i, (start, end) in enumerate(aligned_offsets):
            if start == end:
                continue
            for span_start, span_end in modeling_spans:
                if not (end <= span_start or start >= span_end):
                    mask[i] = 1
                    break
        return mask

    def _build_modeling_mask_from_prefix_decode(self, response_ids: list[int]) -> list[int]:
        """Fallback alignment by decoding token prefixes and mapping char growth per token.

        This is slower but avoids dependence on exact retokenization alignment.
        """
        if not response_ids:
            return []
        decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        modeling_spans = self._find_modeling_char_spans(decoded_text)
        if not modeling_spans:
            return [0] * len(response_ids)
        mask = [0] * len(response_ids)
        prev_text = ''
        for i in range(len(response_ids)):
            cur_text = self.tokenizer.decode(response_ids[:i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            token_char_start = len(prev_text)
            token_char_end = len(cur_text)
            if token_char_end > token_char_start:
                for span_start, span_end in modeling_spans:
                    if not (token_char_end <= span_start or token_char_start >= span_end):
                        mask[i] = 1
                        break
            prev_text = cur_text
        return mask

    def _build_modeling_only_response_mask(self, response_ids: list[int]) -> list[int]:
        """Return response_mask where only tokens inside <modeling>...</modeling> are 1."""
        if not response_ids:
            return []
        decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        mask = self._build_modeling_mask_from_offsets(response_ids, decoded_text)
        if mask is not None:
            return mask
        return self._build_modeling_mask_from_prefix_decode(response_ids)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs['raw_prompt'])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get('images')
        videos = multi_modal_data.get('videos')
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)
        metrics = {}
        with simple_timer('generate_sequences', metrics):
            output: TokenOutput = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=images, video_data=videos)
        if metrics.get('num_preempted') is None:
            metrics['num_preempted'] = output.num_preempted if output.num_preempted is not None else -1
        response_ids = output.token_ids[:self.response_length]
        response_mask = self._build_modeling_only_response_mask(response_ids)
        output: AgentLoopOutput = AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask, response_logprobs=output.log_probs[:self.response_length] if output.log_probs else None, routed_experts=output.routed_experts[:len(prompt_ids) + self.response_length] if output.routed_experts is not None else None, multi_modal_data=multi_modal_data, num_turns=2, metrics=metrics, extra_fields=output.extra_fields)
        output.extra_fields.update({'turn_scores': [], 'tool_rewards': []})
        return output
