# Modified for PhysReasoning: modeling rollout, physics rewards, metrics, or image handling.
from phys_utils.prompt_loader import load_prompt
import asyncio
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from openai import APIConnectionError, APIStatusError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from phys_utils import AsyncLLMClient, judge_one, extract_boxed, TYPE_CLASSIFICATION_PROMPT_TEMPLATE
RUBRIC_JUDGE_PROMPT = load_prompt('verl.verl.utils.reward_score.physics.RUBRIC_JUDGE_PROMPT')
PARSE_TIMEOUT_SEC = 30.0
logger = logging.getLogger(__file__)
Q_TYPE = ['Numerical', 'Symbolic', 'Textual']

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

def _extract_solution_block(text: str) -> Optional[str]:
    return _extract_tag_block(text, 'solution')

def _extract_answer_block(text: str) -> Optional[str]:
    return _extract_tag_block(text, 'answer')

def parse_binary_rating(text: str) -> float:
    text = (text or '').strip()
    m = re.fullmatch('\\[?\\s*([01])(?:\\.0)?\\s*\\]?', text)
    if m:
        return float(m.group(1))
    m = re.search('Rating\\b[\\s\\S]{0,80}?\\[?\\s*([01])(?:\\.0)?\\s*\\]?', text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    logger.warning('[RubricParseFailed] Invalid judge output; content omitted.')
    return 0.0

async def safe_chat_generate(model, backup_model, user_prompt: str, **kwargs):
    try:
        return await model.chat_generate(user_prompt=user_prompt, **kwargs)
    except Exception as e:
        logger.warning(f'[PrimaryModelFailed] fallback to backup_model. Error type: {type(e).__name__}')
        try:
            return await backup_model.chat_generate(user_prompt=user_prompt, **kwargs)
        except Exception as e2:
            logger.warning(f'[BackupModelFailed] return default Rating: 0. Error type: {type(e2).__name__}')
            return ('0.0', {'error': 'both primary and backup model failed'})
default_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6), retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)))

async def compute_score(response: str, ground_truth: str, extra_info: dict, model: AsyncLLMClient, backup_model: AsyncLLMClient, enable_rubric: bool=True, enable_correctness: bool=True, enable_reflection: bool=False, normalize_rubric: bool=True, modeling_execute: bool=False, remote_execute: bool=False, **kwargs) -> dict[str, float]:
    """
    Computes rewards:
    1. Accuracy
    2. Rubric
    3. Format

    response has the following format:
    <modeling>...</modeling>
    <solution>...</solution>
    <answer>...</answer>

    or

    <modeling>...</modeling>
    """
    modeling_part = _extract_modeling_block(response)
    solution_part = _extract_solution_block(response)
    answer_part = _extract_answer_block(response)
    rubric = extra_info['rubric']
    question = extra_info.get('question')
    gt_answer = ground_truth
    q_type = extra_info.get('type', None)
    if q_type is None:
        qtype_content, qtype_meta = await safe_chat_generate(model=model, backup_model=backup_model, user_prompt=TYPE_CLASSIFICATION_PROMPT_TEMPLATE.format(question=question, answer=ground_truth))
        for c in Q_TYPE:
            if c in qtype_content:
                q_type = c
        if q_type is None:
            q_type = 'Textual'
    format_score = 0.0
    if modeling_execute:
        if modeling_part:
            format_score = 1.0
    else:
        if modeling_part:
            format_score += 1.0
        if solution_part:
            format_score += 1.0
        if answer_part:
            format_score += 1.0
        format_score = format_score / 3.0
    correctness_score = 0.0
    if enable_correctness:
        pred_answer = None
        if modeling_execute == False:
            if answer_part:
                pred_answer = extract_boxed(answer_part)
            else:
                pred_answer = extract_boxed(response)
        elif modeling_execute and remote_execute:
            remote_execution = extra_info.get('remote_execution', None)
            if remote_execution:
                pred_answer = extract_boxed(remote_execution)
        else:
            pred_answer = extract_boxed(response)
        if pred_answer:
            result = 0.0
            try:
                result = await judge_one(model=model, gt_answer=gt_answer, pred_answer=pred_answer, q_type=q_type, rtol=0.01)
            except Exception as e1:
                try:
                    result = await judge_one(model=backup_model, gt_answer=gt_answer, pred_answer=pred_answer, q_type=q_type, rtol=0.01)
                except Exception as e2:
                    logger.warning(f'[JudgeFailed] Both primary and backup model failed to judge correctness. Error types: {type(e1).__name__}, {type(e2).__name__}. Return default correctness score 0.0')
                    result = 0.0
            correctness_score = result
    rubric_score = 0.0
    rubric_score_phys = 0.0
    rubric_score_plan = 0.0
    rubric_score_solution = 0.0
    rubric_score_visual = 0.0
    rubric_other_score = 0.0
    total_weight = 0.0
    if enable_rubric and rubric:
        for criterion in rubric:
            total_weight += criterion.get('weight', 1.0)
        for criterion in rubric:
            criterion_description = criterion.get('description', '')
            criterion_weight = float(criterion.get('weight', 1.0))
            criterion_aspect = criterion.get('aspect', '')
            criterion_eval_content, criterion_eval_meta = await safe_chat_generate(model=model, backup_model=backup_model, user_prompt=RUBRIC_JUDGE_PROMPT.replace('__RESPONSE___', response).replace('__CRITERION___', criterion_description))
            rating = parse_binary_rating(criterion_eval_content)
            weighted_rating = rating * (criterion_weight / total_weight) if total_weight > 0 else 0.0
            if criterion_aspect == 'Physical Modeling':
                rubric_score_phys += weighted_rating
            elif criterion_aspect == 'Strategic Planning':
                rubric_score_plan += weighted_rating
            elif criterion_aspect == 'Solution Steps':
                rubric_score_solution += weighted_rating
            elif criterion_aspect == 'Visual Info':
                rubric_score_visual += weighted_rating
            else:
                rubric_other_score += weighted_rating
        rubric_score = rubric_score_phys + rubric_score_plan + rubric_score_solution + rubric_score_visual + rubric_other_score
    reflection_score = 0.0
    if enable_reflection:
        reflection_keywords = ['wait', 'hold on', 'let me check', 'rethink', 'verify']
        t = response.lower()
        reflection_score = 1.0 if any((k in t for k in reflection_keywords)) else 0.0
    rubric_weight = kwargs.get('rubric_weight', 1.0)
    correctness_weight = kwargs.get('correctness_weight', 1.0)
    format_weight = kwargs.get('format_weight', 0.5)
    total_score = rubric_score * rubric_weight + correctness_score * correctness_weight + format_score * format_weight + reflection_score
    logger.info(f'[ComputeScore] Final Total Score: {total_score}')
    out = {'score': total_score, 'rubric_score': rubric_score, 'correctness_score': correctness_score, 'format_score': format_score, 'reflection_score': reflection_score, 'rubric_score_phys': rubric_score_phys, 'rubric_score_plan': rubric_score_plan, 'rubric_score_solution': rubric_score_solution, 'rubric_score_visual': rubric_score_visual, 'rubric_other_score': rubric_other_score}
    return out

def compute_rubric_score(response: str) -> Tuple[float, int, List[Dict[str, Any]]]:
    """
    Parse rubric-evaluator output and resolve a final rubric score.

    Expected format:
        Rubric 1: [1] | Explanation: ...
        Rubric 2: [0] | Explanation: ...
        ...
        Final Score: X/Y
        Total Satisfied: X

    Resolution priority:
        1) If per-rubric lines exist, use their summed ratings as source of truth.
        2) Else if Final Score exists, use (X, Y).
        3) Else if Total Satisfied exists, use (X, X) as a weak fallback.
        4) Else return (0.0, 0).

    Returns:
        raw_score: summed rubric score before normalization
        max_score: maximum possible rubric score
        parsed_scores: parsed rubric item list
    """
    _RUBRIC_LINE_RE = re.compile('^\\s*Rubric\\s+(\\d+)\\s*:\\s*\\[([01])\\]\\s*\\|\\s*Explanation\\s*:\\s*(.*)$', re.IGNORECASE)
    _FINAL_SCORE_RE = re.compile('^\\s*Final\\s*Score\\s*:\\s*(\\d+)\\s*/\\s*(\\d+)\\s*$', re.IGNORECASE)
    _TOTAL_SATISFIED_RE = re.compile('^\\s*Total\\s*Satisfied\\s*:\\s*(\\d+)\\s*$', re.IGNORECASE)
    parsed_scores: List[Dict[str, Any]] = []
    final_score: Optional[int] = None
    final_total: Optional[int] = None
    total_satisfied: Optional[int] = None
    for raw in (response or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _RUBRIC_LINE_RE.match(line)
        if m:
            rubric_idx = int(m.group(1))
            rating = int(m.group(2))
            explanation = m.group(3).strip()
            parsed_scores.append({'idx': rubric_idx, 'rating': rating, 'explanation': explanation})
            continue
        m = _FINAL_SCORE_RE.match(line)
        if m:
            final_score = int(m.group(1))
            final_total = int(m.group(2))
            continue
        m = _TOTAL_SATISFIED_RE.match(line)
        if m:
            total_satisfied = int(m.group(1))
            continue
    if parsed_scores:
        parsed_scores.sort(key=lambda x: x['idx'])
        raw_score = float(sum((item['rating'] for item in parsed_scores)))
        max_score = len(parsed_scores)
        return (raw_score, max_score, parsed_scores)
    if final_score is not None and final_total is not None and (final_total > 0):
        return (float(final_score), int(final_total), parsed_scores)
    if total_satisfied is not None:
        return (float(total_satisfied), int(total_satisfied), parsed_scores)
    return (0.0, 0, parsed_scores)
