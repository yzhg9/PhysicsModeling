from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

class JsonlWriter:

    def __init__(self, path: str, mode: str='w', ensure_ascii: bool=False, fsync: bool=False):
        if 'b' in mode:
            raise ValueError("Use text mode for JSONL (e.g., 'w' or 'a').")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.f = open(path, mode, encoding='utf-8', newline='\n')
        self.ensure_ascii = ensure_ascii
        self.fsync_flag = fsync

    def write(self, obj: Any) -> None:
        json.dump(obj, self.f, ensure_ascii=self.ensure_ascii)
        self.f.write('\n')

    def flush(self) -> None:
        self.f.flush()
        if self.fsync_flag:
            os.fsync(self.f.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.f.close()

    def __enter__(self) -> 'JsonlWriter':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_records(path: str) -> List[Dict[str, Any]]:
    """Load a .json (list/dict) or .jsonl into a list of dict records."""
    if path.endswith('.jsonl'):
        return list(iter_jsonl(path))
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return list(obj.values())
        raise ValueError(f'Unsupported JSON root type: {type(obj)}')
    raise ValueError(f'Unsupported file type: {path}')

def load_existing_ids(output_jsonl: str, id_key: str='id') -> Set[str]:
    """Resume helper: read output jsonl and collect completed ids."""
    if not output_jsonl or not os.path.exists(output_jsonl):
        return set()
    done: Set[str] = set()
    for obj in iter_jsonl(output_jsonl):
        _id = obj.get(id_key)
        if _id is not None:
            done.add(str(_id))
    return done

def load_existing_attempts(output_jsonl: str, id_key: str='id', attempt_key: str='attempt') -> Set[Tuple[str, int]]:
    """Resume helper for best-of-N: collect (id, attempt) pairs."""
    if not output_jsonl or not os.path.exists(output_jsonl):
        return set()
    done: Set[Tuple[str, int]] = set()
    for obj in iter_jsonl(output_jsonl):
        _id = obj.get(id_key)
        _a = obj.get(attempt_key)
        if _id is None or _a is None:
            continue
        try:
            done.add((str(_id), int(_a)))
        except Exception:
            continue
    return done
