import ast
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
_ALLOWED_NAMES = {'pi': math.pi, 'π': math.pi, 'e': math.e}
_ALLOWED_FUNCS = {'sqrt': math.sqrt}
_PI_LATEX_RE = re.compile('\\\\pi\\b')
_TENPOW_RE = re.compile('(?P<a>(?:[+-]?\\d+(?:\\.\\d*)?|\\.\\d+))\\s*(?:[×xX\\*·]\\s*)?10\\s*\\^\\s*(?P<e>[+-]?\\d+)')
_POW_RE = re.compile('(?P<base>(?:[+-]?\\d+(?:\\.\\d*)?|\\.\\d+))\\s*\\^\\s*(?:\\{\\s*(?P<exp1>[+-]?\\d+)\\s*\\}|(?P<exp2>[+-]?\\d+))')

def _extract_braced(s: str, start: int) -> Tuple[str, int]:
    """
    Expect s[start] == '{'.
    Return (content_inside_braces, next_index_after_closing_brace).
    Supports nested braces.
    """
    if start >= len(s) or s[start] != '{':
        raise ValueError(f"Expected '{{' at position {start}")
    depth = 0
    buf = []
    i = start
    while i < len(s):
        ch = s[i]
        if ch == '{':
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return (''.join(buf), i + 1)
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    raise ValueError('Unbalanced braces')

def _replace_frac_once(s: str) -> Tuple[str, bool]:
    idx = s.find('\\frac')
    if idx == -1:
        return (s, False)
    i = idx + len('\\frac')
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] != '{':
        return (s, False)
    try:
        num, j = _extract_braced(s, i)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != '{':
            return (s, False)
        den, k = _extract_braced(s, j)
    except ValueError:
        return (s, False)
    repl = f'(({num})/({den}))'
    return (s[:idx] + repl + s[k:], True)

def _replace_all_fracs(s: str) -> str:
    while True:
        s_new, changed = _replace_frac_once(s)
        if not changed:
            return s
        s = s_new

def _replace_sqrt_once(s: str) -> Tuple[str, bool]:
    idx = s.find('\\sqrt')
    if idx == -1:
        return (s, False)
    i = idx + len('\\sqrt')
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] != '{':
        return (s, False)
    try:
        inner, j = _extract_braced(s, i)
    except ValueError:
        return (s, False)
    repl = f'sqrt({inner})'
    return (s[:idx] + repl + s[j:], True)

def _replace_all_sqrts(s: str) -> str:
    while True:
        s_new, changed = _replace_sqrt_once(s)
        if not changed:
            return s
        s = s_new

def _normalize_number_text(s: str) -> str:
    s = s.strip()
    s = s.replace('−', '-').replace('–', '-').replace('—', '-')
    s = s.replace('×', '*').replace('·', '*')
    s = s.replace('π', 'pi')
    s = s.replace('\\times', '*')
    s = re.sub('(?<=\\d),(?=\\d{3}(\\D|$))', '', s)
    s = _PI_LATEX_RE.sub('pi', s)
    s = re.sub('10\\s*\\^\\s*\\{\\s*([+-]?\\d+)\\s*\\}', '10^\\1', s)
    s = re.sub('10\\s*\\^\\s*\\(\\s*([+-]?\\d+)\\s*\\)', '10^\\1', s)
    s = _replace_all_fracs(s)
    s = _replace_all_sqrts(s)
    s = _TENPOW_RE.sub(lambda m: f"({m.group('a')}*10**({m.group('e')}))", s)
    s = _POW_RE.sub(lambda m: f"({m.group('base')}**({m.group('exp1') or m.group('exp2')}))", s)
    s = re.sub('(?<=\\d)\\s*(?=pi\\b)', '*', s)
    s = re.sub('(?<=\\))\\s*(?=pi\\b)', '*', s)
    s = re.sub('(?<=\\d)\\s*(?=sqrt\\()', '*', s)
    s = re.sub('(?<=\\))\\s*(?=sqrt\\()', '*', s)
    s = re.sub('\\s+', '', s)
    return s

class _SafeEval(ast.NodeVisitor):

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError('bad constant')
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError('bad binop')
        if isinstance(node, ast.UnaryOp):
            val = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError('bad unary')
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                return float(_ALLOWED_NAMES[node.id])
            raise ValueError('bad name')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError('bad call')
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError('bad func')
            if len(node.args) != 1:
                raise ValueError('bad arity')
            arg0 = self.visit(node.args[0])
            return float(_ALLOWED_FUNCS[fname](arg0))
        raise ValueError('unsupported expr')

def parse_number_fallback(num_str: str) -> float:
    """
    Try float() first; if fails, parse a restricted math expression safely.
    Supports: fractions, pi, sqrt, scientific variants like 1.2×10^-3, 10^3.
    """
    s = num_str.strip()
    try:
        return float(s)
    except Exception:
        pass
    s2 = _normalize_number_text(s)
    try:
        return float(s2)
    except Exception:
        pass
    tree = ast.parse(s2, mode='eval')
    return _SafeEval().visit(tree)
NUM_UNIT_LOOSE_RE = re.compile('^\\s*(?P<num>.+?)\\s*(?P<unit>\\[[^\\]]*\\])?\\s*$')

def split_value_unit(s: str) -> Tuple[float, str]:
    s = (s or '').strip()
    m = NUM_UNIT_LOOSE_RE.match(s)
    if not m:
        raise ValueError(f'Cannot parse value+unit from: {s!r}')
    num = (m.group('num') or '').strip()
    unit = (m.group('unit') or '[]').strip()
    value = parse_number_fallback(num)
    return (value, unit)

def split_many_value_unit(text: str) -> List[Tuple[float, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(split_value_unit(line))
    return out
Pair = Tuple[float, str]

def almost_equal(a: float, b: float, rtol: float=0.005, atol: float=0) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
Dim = Dict[str, int]

def _add_dim(a: Dim, b: Dim, k: int=1) -> Dim:
    out = dict(a)
    for key, exp in b.items():
        out[key] = out.get(key, 0) + k * exp
        if out[key] == 0:
            del out[key]
    return out

def _mul_dim(a: Dim, b: Dim) -> Dim:
    return _add_dim(a, b, 1)

def _div_dim(a: Dim, b: Dim) -> Dim:
    return _add_dim(a, b, -1)

def _pow_dim(a: Dim, p: int) -> Dim:
    if p == 0:
        return {}
    return {k: v * p for k, v in a.items() if v * p != 0}
BASE: Dict[str, Dim] = {'kg': {'kg': 1}, 'm': {'m': 1}, 's': {'s': 1}, 'A': {'A': 1}, 'K': {'K': 1}, 'mol': {'mol': 1}, 'cd': {'cd': 1}, 'rad': {}, 'sr': {}}
DERIVED: Dict[str, Dim] = {}
DERIVED['Hz'] = _pow_dim(BASE['s'], -1)
DERIVED['N'] = _div_dim(_mul_dim(BASE['kg'], BASE['m']), _pow_dim(BASE['s'], 2))
DERIVED['Pa'] = _div_dim(DERIVED['N'], _pow_dim(BASE['m'], 2))
DERIVED['J'] = _mul_dim(DERIVED['N'], BASE['m'])
DERIVED['W'] = _div_dim(DERIVED['J'], BASE['s'])
DERIVED['C'] = _mul_dim(BASE['A'], BASE['s'])
DERIVED['V'] = _div_dim(DERIVED['W'], BASE['A'])
DERIVED['Ohm'] = _div_dim(DERIVED['V'], BASE['A'])
DERIVED['S'] = _pow_dim(DERIVED['Ohm'], -1)
DERIVED['F'] = _div_dim(DERIVED['C'], DERIVED['V'])
DERIVED['Wb'] = _mul_dim(DERIVED['V'], BASE['s'])
DERIVED['T'] = _div_dim(DERIVED['Wb'], _pow_dim(BASE['m'], 2))
DERIVED['lm'] = {'cd': 1}
DERIVED['lx'] = {'cd': 1, 'm': -2}
UNIT_TABLE: Dict[str, Dim] = {**BASE, **DERIVED}
_FACTOR_RE = re.compile('(?P<name>[A-Za-zμ°]+)(?:\\^(?P<pow>[-+]?\\d+))?$')

def _strip_brackets(u: str) -> str:
    u = (u or '').strip()
    if u == '[]':
        return ''
    if u.startswith('[') and u.endswith(']'):
        return u[1:-1].strip()
    return u.strip()
SI_PREFIX: Dict[str, float] = {'Y': 1e+24, 'Z': 1e+21, 'E': 1e+18, 'P': 1000000000000000.0, 'T': 1000000000000.0, 'G': 1000000000.0, 'M': 1000000.0, 'k': 1000.0, 'h': 100.0, 'da': 10.0, 'd': 0.1, 'c': 0.01, 'm': 0.001, 'u': 1e-06, 'μ': 1e-06, 'n': 1e-09, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24}
EV_TO_J = 1.602176634e-19
DEG_TO_RAD = math.pi / 180.0
SPECIAL_UNIT_FACTOR: Dict[str, Tuple[Dim, float]] = {'eV': (DERIVED['J'], EV_TO_J), 'deg': ({}, DEG_TO_RAD), 'degree': ({}, DEG_TO_RAD), 'circ': ({}, DEG_TO_RAD), '°': ({}, DEG_TO_RAD), 'min': (BASE['s'], 60.0), 'minute': (BASE['s'], 60.0), 'mins': (BASE['s'], 60.0), 'minutes': (BASE['s'], 60.0), 'hr': (BASE['s'], 3600.0), 'hour': (BASE['s'], 3600.0), 'hrs': (BASE['s'], 3600.0), 'hours': (BASE['s'], 3600.0), 'rpm': (DERIVED['Hz'], 1.0 / 60.0), 'bar': (DERIVED['Pa'], 100000.0), 'atm': (DERIVED['Pa'], 101325.0), 'mmHg': (DERIVED['Pa'], 133.322387415), 'cal': (DERIVED['J'], 4.184), 'kcal': (DERIVED['J'], 4184.0)}

def _normalize_unit_inner(inner: str) -> str:
    inner = inner.replace('Ω', 'Ohm')
    inner = inner.replace('\\cdot', '*').replace('·', '*')
    inner = inner.replace('\\,', ' ').replace('\xa0', ' ')
    inner = re.sub('\\s+', ' ', inner).strip()
    inner = re.sub('\\s*([*/])\\s*', '\\1', inner)
    inner = re.sub('(?<=[A-Za-z0-9\\)\\]])\\s+(?=[A-Za-z0-9\\(\\[])', '*', inner)
    inner = inner.replace(' ', '')
    inner = re.sub('\\*{2,}', '*', inner)
    return inner

@lru_cache(maxsize=4096)
def unit_to_dim_factor(unit_bracket: str) -> Tuple[Dim, float]:
    inner = _strip_brackets(unit_bracket)
    if inner == '':
        return ({}, 1.0)
    inner = _normalize_unit_inner(inner)
    parts = re.split('([*/])', inner)
    dim: Dim = {}
    factor: float = 1.0
    op = '*'

    def resolve_token(name: str) -> Tuple[Dim, float]:
        if name in SPECIAL_UNIT_FACTOR:
            return SPECIAL_UNIT_FACTOR[name]
        if name in UNIT_TABLE:
            return (UNIT_TABLE[name], 1.0)
        if name == 'g':
            return (BASE['kg'], 0.001)
        if name.startswith('da') and len(name) > 2:
            base = name[2:]
            if base in UNIT_TABLE:
                return (UNIT_TABLE[base], SI_PREFIX['da'])
            if base in SPECIAL_UNIT_FACTOR:
                d, f = SPECIAL_UNIT_FACTOR[base]
                return (d, SI_PREFIX['da'] * f)
            if base == 'g':
                return (BASE['kg'], SI_PREFIX['da'] * 0.001)
        if len(name) >= 2 and name[0] in SI_PREFIX:
            pref = name[0]
            base = name[1:]
            if base in UNIT_TABLE:
                return (UNIT_TABLE[base], SI_PREFIX[pref])
            if base in SPECIAL_UNIT_FACTOR:
                d, f = SPECIAL_UNIT_FACTOR[base]
                return (d, SI_PREFIX[pref] * f)
            if base == 'g':
                return (BASE['kg'], SI_PREFIX[pref] * 0.001)
        raise ValueError(f'Unknown unit token: {name!r} in {unit_bracket!r}')
    for part in parts:
        if part in ('*', '/'):
            op = part
            continue
        if not part:
            continue
        m = _FACTOR_RE.fullmatch(part)
        if not m:
            raise ValueError(f'Unsupported unit factor: {part!r} in {unit_bracket!r}')
        name = m.group('name')
        p = int(m.group('pow') or '1')
        token_dim, token_factor = resolve_token(name)
        token_dim = _pow_dim(token_dim, p)
        token_factor = token_factor ** p
        if op == '*':
            dim = _mul_dim(dim, token_dim)
            factor *= token_factor
        else:
            dim = _div_dim(dim, token_dim)
            factor /= token_factor
    return (dim, factor)

def units_match_fast_then_equiv(u_pred: str, u_gold: str) -> bool:
    up = (u_pred or '').strip()
    ug = (u_gold or '').strip()
    if up == ug:
        return True
    try:
        dp, _ = unit_to_dim_factor(up)
        dg, _ = unit_to_dim_factor(ug)
        return dp == dg
    except ValueError:
        return False

def is_missing_unit(u: str) -> bool:
    u = (u or '').strip()
    return u == '' or u == '[]'

def score_numerical(gold: List[Pair], pred: List[Pair], *, rtol: float=0.005, atol: float=0, total_mode: str='gold', allow_unitless_numeric_match: bool=True) -> Dict[str, Any]:
    if total_mode not in ('gold', 'max'):
        raise ValueError("total_mode must be 'gold' or 'max'")
    used = [False] * len(gold)
    matches = []
    misses = []
    correct = 0
    for pv, pu in pred:
        best_j: Optional[int] = None
        best_diff = float('inf')
        best_info: Optional[Dict[str, Any]] = None
        for j, (gv, gu) in enumerate(gold):
            if used[j]:
                continue
            pu_missing = is_missing_unit(pu)
            gu_missing = is_missing_unit(gu)
            pv_cmp, gv_cmp = (pv, gv)
            if pu_missing or gu_missing:
                if not allow_unitless_numeric_match:
                    continue
            elif (pu or '').strip() == (gu or '').strip():
                pv_cmp, gv_cmp = (pv, gv)
            else:
                try:
                    dp, fp = unit_to_dim_factor(pu)
                    dg, fg = unit_to_dim_factor(gu)
                except ValueError:
                    continue
                if dp != dg:
                    continue
                pv_cmp = pv * fp
                gv_cmp = gv * fg
            if not almost_equal(pv_cmp, gv_cmp, rtol=rtol, atol=atol):
                continue
            diff = abs(pv_cmp - gv_cmp)
            if diff < best_diff:
                best_diff = diff
                best_j = j
                best_info = {'pred': (pv, pu), 'gold': (gv, gu), 'pred_scaled': pv_cmp, 'gold_scaled': gv_cmp, 'abs_diff_scaled': diff}
        if best_j is None:
            reason = 'no_match'
            for j, (gv, gu) in enumerate(gold):
                if used[j]:
                    continue
                if almost_equal(pv, gv, rtol=rtol, atol=atol):
                    if not (is_missing_unit(pu) or is_missing_unit(gu)):
                        if not units_match_fast_then_equiv(pu, gu):
                            reason = 'unit_mismatch'
                            break
            misses.append({'pred': (pv, pu), 'reason': reason})
        else:
            used[best_j] = True
            correct += 1
            matches.append(best_info if best_info is not None else {'pred': (pv, pu), 'gold': gold[best_j]})
    unmatched_gold = [gold[j] for j in range(len(gold)) if not used[j]]
    denom = len(gold) if total_mode == 'gold' else max(len(pred), len(gold))
    acc = correct / denom if denom > 0 else 0.0
    return {'correct': correct, 'total': denom, 'accuracy': acc, 'matches': matches, 'misses': misses, 'unmatched_gold': unmatched_gold, 'len_gold': len(gold), 'len_pred': len(pred), 'allow_unitless_numeric_match': allow_unitless_numeric_match}

def fix_fracs(s: str) -> str:
    """
    Normalize malformed LaTeX fractions:
      \\frac12 -> \\frac{1}{2}
      \\frac1b -> \\frac{1}{b}
    """
    substrs = s.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if not substr:
                return s
            if substr[0] == '{':
                new_str += substr
            else:
                if len(substr) < 2:
                    return s
                a = substr[0]
                b = substr[1]
                if b != '{':
                    post_substr = substr[2:] if len(substr) > 2 else ''
                    new_str += '{' + a + '}{' + b + '}' + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ''
                    new_str += '{' + a + '}' + b + post_substr
    return new_str

def fix_sqrt(s: str) -> str:
    """
    Normalize malformed sqrt:
      \\sqrt2 -> \\sqrt{2}
    """
    if '\\sqrt' not in s:
        return s
    splits = s.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += '\\sqrt'
            continue
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def _strip_simple_latex_wrapper(s: str, name: str) -> str:
    pat = f'\\\\{name}\\s*\\{{(.+)\\}}'
    m = re.fullmatch(pat, s)
    return m.group(1).strip() if m else s

def _looks_non_numeric_answer(s: str) -> bool:
    s_low = s.lower()
    bad_phrases = ['cannot be determined', "can't be determined", 'cannot determine', 'not enough information', 'insufficient information', 'no solution', 'undefined', 'impossible', 'cannot solve', 'cannot compute']
    return any((p in s_low for p in bad_phrases))

def _strip_answer_prefix(s: str) -> str:
    s = s.strip()
    s = re.sub('^\\s*(?:the\\s+answer|answer|final\\s+answer|result|value)\\s*[:=]\\s*', '', s, flags=re.I)
    s = re.sub('^\\s*[A-Za-z][A-Za-z0-9_]*(?:\\{[^{}]*\\})?\\s*=\\s*', '', s)
    return s.strip()

def _preclean_answer_text(s: str) -> str:
    """
    Clean boxed/raw final answer text while preserving unit meaning.
    """
    s = str(s).strip()
    s = s.replace('\n', ' ')
    s = s.replace('\\\\', '\\')
    s = s.replace('\\!', '')
    s = s.replace('\\,', ' ')
    s = s.replace('\\;', ' ')
    s = s.replace('~', ' ')
    s = s.replace('−', '-').replace('–', '-').replace('—', '-')
    s = s.replace('Ω', 'Ohm')
    s = s.replace('°', ' deg ')
    s = s.replace('×', ' \\times ')
    s = s.replace('·', ' \\cdot ')
    s = s.replace('tfrac', 'frac')
    s = s.replace('dfrac', 'frac')
    s = s.replace('\\left', '')
    s = s.replace('\\right', '')
    s = s.replace('\\$', '')
    if len(s) >= 2 and s.startswith('$') and s.endswith('$'):
        s = s[1:-1].strip()
    s = _strip_simple_latex_wrapper(s, 'boxed')
    s = _strip_simple_latex_wrapper(s, 'mathbf')
    s = _strip_simple_latex_wrapper(s, 'mathrm')
    s = _strip_simple_latex_wrapper(s, 'textrm')
    s = fix_sqrt(s)
    s = fix_fracs(s)
    s = re.sub('\\^\\s*\\{\\\\circ\\}', ' deg', s)
    s = re.sub('\\^\\s*\\\\circ\\b', ' deg', s)
    s = re.sub('\\\\circ\\b', ' deg', s)
    s = re.sub('\\\\text\\s*\\{([^{}]*)\\}', ' \\1 ', s)
    s = re.sub('\\\\mathrm\\s*\\{([^{}]*)\\}', ' \\1 ', s)
    s = s.replace('\\%', '%')
    s = s.replace(' .', ' 0.')
    s = s.replace('{.', '{0.')
    if s.startswith('.'):
        s = '0' + s
    s = re.sub('\\s+', ' ', s).strip()
    return s

def _split_answer_candidates(s: str) -> List[str]:
    s = re.sub('(?<=\\d),(?=\\d{3}(\\D|$))', '', s)
    parts = re.split('\\s*(?:,|;|\\band\\b|\\n)\\s*', s)
    return [p.strip() for p in parts if p.strip()]
_NUM_PREFIX_RE = re.compile('\n    ^\\s*\n    (?P<num>\n        [+-]?\n        (?:\n            \\\\frac\n            |\n            \\\\sqrt\n            |\n            (?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?\n                (?:\\s*\\\\times\\s*10\\s*\\^\\s*\\{?[+-]?\\d+\\}?)?\n            |\n            \\d+\\s*/\\s*\\d+\n            |\n            pi|\\\\pi|π\n        )\n        .*?\n    )\n    \\s*(?P<unit>.*?)\\s*$\n    ', re.X)

def normalize_unit_for_output(unit_str: str) -> Optional[str]:
    """
    Normalize raw unit text into output form without brackets.
    Accepts both:
      'J*s'
      '[J*s]'
    Returns normalized inner unit string, e.g. 'J*s', 'rad', ''.
    """
    s = (unit_str or '').strip()
    if not s:
        return ''
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    s = re.sub('\\\\text\\s*\\{([^{}]*)\\}', '\\1', s)
    s = re.sub('\\\\mathrm\\s*\\{([^{}]*)\\}', '\\1', s)
    s = s.replace('\\,', ' ').replace('\\;', ' ').replace('\\!', ' ')
    s = s.replace('\\cdot', '*').replace('·', '*')
    s = s.replace('Ω', 'Ohm')
    s = s.replace('°', 'deg')
    s = s.replace('{', '').replace('}', '')
    s = re.sub('\\s+', ' ', s).strip()
    s = re.sub('\\^\\s*deg\\b', 'deg', s)
    s = s.replace(' per ', '/')
    s = re.sub('\\s*/\\s*', '/', s)
    s = re.sub('\\s*\\*\\s*', '*', s)
    s = re.sub('(?<=[A-Za-z])\\s+(?=[A-Za-z])', '*', s)
    s = s.replace(' ', '')
    alias = {'sec': 's', 'second': 's', 'seconds': 's', 'meter': 'm', 'meters': 'm', 'degree': 'deg', 'degrees': 'deg', 'ohm': 'Ohm'}
    if s.lower() in alias:
        s = alias[s.lower()]
    if s in {'', '[]'}:
        return ''
    try:
        unit_to_dim_factor(f'[{s}]')
    except Exception:
        return None
    return s

def _split_num_and_unit_tail(s: str) -> Optional[Tuple[str, str]]:
    """
    Heuristic splitter for strings like:
      6.58e-34J*s
      3.141592653589793 rad
      \x0crac{8^{2}}{5}m/s
      x=12
    Returns (num_str, unit_str) or None.
    """
    s = s.strip()
    if not s:
        return None
    m_norm = re.fullmatch('\\s*(.+?)\\s*\\[([^\\]]*)\\]\\s*$', s)
    if m_norm:
        return (m_norm.group(1).strip(), m_norm.group(2).strip())
    for i in range(len(s), 0, -1):
        left = s[:i].strip()
        right = s[i:].strip()
        if not left:
            continue
        try:
            parse_number_fallback(left)
            return (left, right)
        except Exception:
            continue
    return None

def _parse_one_answer_candidate(part: str) -> Optional[Tuple[float, str]]:
    s = part.strip()
    if not s:
        return None
    s = _strip_answer_prefix(s)
    if re.search('\\\\pm|\\bor\\b|either', s, flags=re.I):
        return None
    split_res = _split_num_and_unit_tail(s)
    if split_res is None:
        return None
    num_str, unit_str = split_res
    try:
        value = parse_number_fallback(num_str)
    except Exception:
        return None
    unit = normalize_unit_for_output(unit_str)
    if unit is None:
        return None
    return (value, unit)

def _format_number(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        raise ValueError('invalid numeric result')
    s = f'{x:.15g}'
    s = s.replace('e+0', 'e').replace('e+', 'e')
    return s

def normalize_boxed_numerical_answer(raw: str) -> str:
    """
    Convert boxed/raw answer string to normalized lines:
      <number> [<unit>]
    or FAIL
    """
    if raw is None:
        return 'FAIL'
    s = _preclean_answer_text(raw)
    if not s or _looks_non_numeric_answer(s):
        return 'FAIL'
    normalized = []
    for part in _split_answer_candidates(s):
        item = _parse_one_answer_candidate(part)
        if item is None:
            continue
        value, unit = item
        normalized.append(f'{_format_number(value)} [{unit}]')
    if not normalized:
        return 'FAIL'
    return '\n'.join(normalized)

def judge_numerical_answer(gold_raw: str, pred_raw: str, *, rtol: float=0.005, atol: float=0.0, total_mode: str='gold', allow_unitless_numeric_match: bool=True) -> Dict[str, Any]:
    gold_norm = normalize_boxed_numerical_answer(gold_raw)
    pred_norm = normalize_boxed_numerical_answer(pred_raw)
    if gold_norm == 'FAIL' or pred_norm == 'FAIL':
        return {'correct': 0, 'total': 1, 'accuracy': 0.0, 'gold_norm': gold_norm, 'pred_norm': pred_norm, 'reason': 'normalization_fail'}
    gold = split_many_value_unit(gold_norm)
    pred = split_many_value_unit(pred_norm)
    res = score_numerical(gold, pred, rtol=rtol, atol=atol, total_mode=total_mode, allow_unitless_numeric_match=allow_unitless_numeric_match)
    res['gold_norm'] = gold_norm
    res['pred_norm'] = pred_norm
    return res
