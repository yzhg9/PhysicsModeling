from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

def dedup_records(records: Sequence[Dict[str, Any]], id_key: str='id') -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        _id = r.get(id_key)
        if _id is None:
            out.append(r)
            continue
        sid = str(_id)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(r)
    return out

def build_todo(records: Sequence[Dict[str, Any]], done_ids: Optional[Set[str]]=None, id_key: str='id') -> List[Dict[str, Any]]:
    if not done_ids:
        return list(records)
    todo: List[Dict[str, Any]] = []
    for r in records:
        _id = r.get(id_key)
        if _id is None or str(_id) not in done_ids:
            todo.append(r)
    return todo

async def run_async_map(items: Sequence[Any], worker: Callable[[Any], Awaitable[Any]], on_result: Optional[Callable[[Any], None]]=None, max_in_flight: int=64) -> List[Any]:
    """Run async worker over items with bounded in-flight tasks."""
    sem = asyncio.Semaphore(max_in_flight)
    results: List[Any] = []

    async def _wrap(x: Any) -> Any:
        async with sem:
            return await worker(x)
    tasks = [asyncio.create_task(_wrap(x)) for x in items]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        if on_result is not None:
            on_result(r)
    return results
