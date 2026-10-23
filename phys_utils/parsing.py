from __future__ import annotations
import re
from typing import Dict, List, Optional, Union

def extract_boxed(text: str, take: str='last') -> Union[None, str, List[str]]:
    """Extract LaTeX \x08oxed{...} contents. take in {'first','last','all'}."""
    boxes: List[str] = []
    i, n = (0, len(text))
    key = '\\boxed'
    while i < n:
        j = text.find(key, i)
        if j == -1:
            break
        k = j + len(key)
        while k < n and text[k].isspace():
            k += 1
        if k >= n or text[k] != '{':
            i = k
            continue
        depth = 0
        p = k
        while p < n:
            if text[p] == '{':
                depth += 1
            elif text[p] == '}':
                depth -= 1
                if depth == 0:
                    boxes.append(text[k + 1:p].strip())
                    i = p + 1
                    break
            p += 1
        else:
            i = n
    if not boxes:
        return None if take != 'all' else []
    if take == 'first':
        return boxes[0]
    if take == 'last':
        return boxes[-1]
    return boxes

def extract_judge(text: str) -> Optional[int]:
    """Extract first 0/1 after 'Judgement'/'Judgment'."""
    m = re.search('Judg[e]?ment[^0-9]*([01])\\b', text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def split_blocks(text: str) -> Dict[str, str]:
    """Split text into [Title] blocks (markdown-ish)."""
    pattern = re.compile('^\\[(?P<title>[^\\]]+)\\]\\s*:?\\s*\\n?(?P<body>.*?)(?=^\\[[^\\]]+\\]\\s*:?.*$|\\Z)', re.S | re.M)
    out: Dict[str, str] = {}
    for m in pattern.finditer(text):
        title = m.group('title').strip()
        body = (m.group('body') or '').strip()
        out[title] = body
    return out

def strip_code_fences(text: str) -> str:
    """Remove triple backtick fences if the model wraps a JSON or answer block."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith('```'):
            lines = lines[1:]
            if lines and lines[-1].strip().startswith('```'):
                lines = lines[:-1]
        return '\n'.join(lines).strip()
    return text
