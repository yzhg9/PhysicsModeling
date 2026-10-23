import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from latex2sympy2_extended.sets import FiniteSet
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig
from math_verify.parser import LatexExtractionConfig
WRAP_CMDS = {'mathrm', 'mathbf', 'mathit', 'mathsf', 'mathtt', 'mathcal', 'boldsymbol', 'text', 'textrm', 'textbf', 'textit', 'textsf', 'texttt'}
DROP_CMDS = {',', ';', ':', '!', 'quad', 'qquad', 'enspace'}
DROP_TOKENS = {'\\[', '\\]', '\\(', '\\)'}

def _read_group(s: str, i: int) -> Tuple[Optional[str], int]:
    if i >= len(s) or s[i] != '{':
        return (None, i)
    depth = 1
    start = i + 1
    i += 1
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return (s[start:i], i + 1)
        i += 1
    return (None, start - 1)

def _read_command_name(s: str, i: int) -> Tuple[str, int]:
    assert s[i] == '\\'
    i += 1
    if i >= len(s):
        return ('', i)
    if s[i].isalpha():
        j = i
        while j < len(s) and s[j].isalpha():
            j += 1
        return (s[i:j], j)
    return (s[i], i + 1)

def strip_latex_formatting(expr: str, *, force_backslash: bool=True) -> str:
    dollar_ok = expr.count('$') == 2

    def _strip(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '{':
                inner, j = _read_group(s, i)
                if inner is None:
                    out.append(ch)
                    i += 1
                else:
                    out.append('{')
                    out.append(_strip(inner))
                    out.append('}')
                    i = j
                continue
            if ch == '\\':
                if i + 1 < len(s) and s[i:i + 2] in DROP_TOKENS:
                    i += 2
                    continue
                cmd, j = _read_command_name(s, i)
                if cmd == '':
                    i = j
                    continue
                if cmd in DROP_CMDS:
                    i = j
                    continue
                if cmd in {'left', 'right'}:
                    k = j
                    while k < len(s) and s[k].isspace():
                        k += 1
                    if k < len(s):
                        if s[k] in '()[]|.{}<>':
                            i = k + 1
                            continue
                        if s[k] == '\\':
                            _, k2 = _read_command_name(s, k)
                            i = k2
                            continue
                    i = j
                    continue
                if cmd in WRAP_CMDS:
                    k = j
                    while k < len(s) and s[k].isspace():
                        k += 1
                    inner, k2 = _read_group(s, k)
                    if inner is not None:
                        out.append(_strip(inner))
                        i = k2
                    else:
                        i = j
                    continue
                out.append('\\')
                out.append(cmd)
                i = j
                continue
            out.append(ch)
            i += 1
        res = ''.join(out)
        res = ' '.join(res.split())
        return res
    res = _strip(expr)
    res = res.replace('and', ',')
    res = res.replace('e_', 'e').replace('E_', 'E')
    res = res.replace('_e', '_{e}').replace('_E', '_{E}')
    if dollar_ok:
        pass
    else:
        res = res.replace('$', '')
        res = f'${res}$'
    if force_backslash:
        res = re.sub('\\\\+', '\\\\', res)
    return res

def score_symbolic(gold: List[Any], pred: List[Any], *, total_mode: str='gold') -> Dict[str, Any]:
    if total_mode not in ('gold', 'max'):
        raise ValueError("total_mode must be 'gold' or 'max'")
    used_gold = [False] * len(gold)
    matches = []
    misses = []
    correct = 0
    for p in pred:
        hit_gi = None
        for gi, g in enumerate(gold):
            if used_gold[gi]:
                continue
            if verify(g, p):
                hit_gi = gi
                break
        if hit_gi is None:
            misses.append({'pred': p, 'reason': 'no_match'})
        else:
            used_gold[hit_gi] = True
            correct += 1
            matches.append({'pred': p, 'gold': gold[hit_gi]})
    unmatched_gold = [gold[gi] for gi, used in enumerate(used_gold) if not used]
    denom = len(gold) if total_mode == 'gold' else max(len(gold), len(pred))
    acc = correct / denom if denom > 0 else 0.0
    return {'correct': correct, 'total': denom, 'accuracy': acc, 'matches': matches, 'misses': misses, 'unmatched_gold': unmatched_gold, 'len_gold': len(gold), 'len_pred': len(pred)}
