from __future__ import annotations
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from .llm_client import AsyncLLMClient
from .numerical import judge_numerical_answer
from .symbolic import strip_latex_formatting, score_symbolic
from .prompts import FALLBACK_JUDGE_PROMPT
from latex2sympy2_extended import NormalizationConfig
from math_verify.parser import LatexExtractionConfig
from math_verify import parse
from latex2sympy2_extended.sets import FiniteSet
TYPE_LIST = ['Numerical', 'Symbolic', 'Textual', 'Multi-Choice']

def extract_option_letter(text):
    if not text:
        return None
    text = str(text).strip()
    patterns = ['(?:choose|pick|select)\\s*[:\\-]?\\s*([A-E])', '(?:answer|ans)\\s*[:=\\-]?\\s*([A-E])', '(?:correct answer is|correct is|is)\\s*([A-E])', '\\(([A-E])\\)', '^\\s*([A-E])[\\.\\)]', '^\\s*([A-E])\\s*$']
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search('\\b([A-E])\\b', text.upper())
    if match:
        return match.group(1)
    return None

def rhs_only(s: str) -> str:
    """
    If the string contains '=', keep only the right-hand side.
    - Uses the last '=' to handle chains like 'a=b=c' -> 'c'
    - If RHS becomes empty, return original string.
    """
    if not isinstance(s, str):
        return s
    if '=' not in s:
        return s
    rhs = s.split('=')[-1].strip()
    return rhs if rhs else s

async def fallback_score(model, gt_answer, pred_answer):
    fall_back_prompt = FALLBACK_JUDGE_PROMPT.format(GT=gt_answer, PRED=pred_answer)
    score_content, meta = await model.chat_generate(user_prompt=fall_back_prompt)
    s = (score_content or '').strip()
    m = re.search('Judgment:\\s*([01])\\b', s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.fullmatch('[01]', s)
    if m:
        return float(s)
    m = re.search('([01])\\b\\s*$', s)
    if m:
        return float(m.group(1))
    return 0.0

async def judge_one(model: AsyncLLMClient, gt_answer: str, pred_answer: str, q_type: str=None, rtol: Optional[float]=0.03):
    score = 0.0
    if q_type == 'Numerical':
        try:
            result = judge_numerical_answer(gold_raw=gt_answer, pred_raw=pred_answer, rtol=rtol)
            score = result['accuracy']
            if result['pred_norm'] == 'FAIL' or result['gold_norm'] == 'FAIL':
                raise ValueError('[Numerical] Rule-Based Normalization Fail')
        except Exception as e:
            score = await fallback_score(model=model, gt_answer=gt_answer, pred_answer=pred_answer)
    elif q_type == 'Symbolic':

        async def try_rule_symbolic(gt_ans: str, pred_ans: str) -> float:
            gt_strip = strip_latex_formatting(gt_ans)
            gt_parsed = parse(gt_strip, (LatexExtractionConfig(normalization_config=NormalizationConfig(units=False)),))
            pred_strip = strip_latex_formatting(pred_ans)
            pred_parsed = parse(pred_strip, (LatexExtractionConfig(normalization_config=NormalizationConfig(units=False)),))
            if not (len(gt_parsed) == 2 and len(pred_parsed) == 2):
                raise ValueError('[Symbolic] Parse format unexpected; fallback needed.')
            gt = gt_parsed[0]
            pred = pred_parsed[0]
            if type(gt) is FiniteSet:
                gt_list = list(gt)
            else:
                gt_list = [gt]
            if type(pred) is FiniteSet:
                pred_list = list(pred)
            else:
                pred_list = [pred]
            result = score_symbolic(gt_list, pred_list)
            score_local = float(result.get('accuracy', 0.0))
            if score_local == 0.0:
                raise ValueError('[Symbolic] Rule-Based got 0 score.')
            return score_local
        try:
            score = await try_rule_symbolic(gt_answer, pred_answer)
        except Exception as e1:
            gt_rhs = rhs_only(gt_answer)
            pred_rhs = rhs_only(pred_answer)
            do_retry = gt_rhs != gt_answer or pred_rhs != pred_answer
            if do_retry:
                try:
                    score = await try_rule_symbolic(gt_rhs, pred_rhs)
                except Exception as e2:
                    score = await fallback_score(model=model, gt_answer=gt_answer, pred_answer=pred_answer)
            else:
                score = await fallback_score(model=model, gt_answer=gt_answer, pred_answer=pred_answer)
    elif q_type == 'Multi-Choice':
        gt = extract_option_letter(gt_answer)
        pred = extract_option_letter(pred_answer)
        if gt == pred:
            return 1.0
        else:
            return 0.0
    else:
        score = await fallback_score(model=model, gt_answer=gt_answer, pred_answer=pred_answer)
    return score
