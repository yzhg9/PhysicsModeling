from phys_utils.prompt_loader import load_prompt
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm.asyncio import tqdm_asyncio
from phys_utils.io import JsonlWriter, load_records
from phys_utils.llm_client import AsyncLLMClient
from phys_utils.parsing import extract_boxed
from phys_utils.judge import judge_one
from phys_utils.numerical import split_many_value_unit
SOLVE_WITH_MODELING_PROMPT_TEMPLATE = load_prompt('scripts.solve_with_modeling.SOLVE_WITH_MODELING_PROMPT_TEMPLATE')

def safe_str(x: Any) -> str:
    return '' if x is None else str(x)

def safe_int(x: Any, default: int=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

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

def load_type_info(type_path: Optional[str]) -> Dict[str, Dict[str, Any]]:

    if not type_path or not os.path.exists(type_path):
        return {}
    info: Dict[str, Dict[str, Any]] = {}

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
    try:
        objs = load_records(type_path)
    except Exception:
        objs = None
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

def choose_gt_answer(rec: Dict[str, Any], q_type: str, type_info: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[str]]:

    raw_gt = (rec.get('answer') or rec.get('gt_answer') or '').strip()
    rec_norm = (rec.get('normalized_answer') or '').strip()
    typefile_norm = get_typefile_normalized_answer(rec, type_info)
    if typefile_norm is not None and typefile_norm.strip().lower() != 'fail':
        if q_type == 'Numerical':
            return (split_many_value_unit(typefile_norm), typefile_norm)
        return (typefile_norm, typefile_norm)
    if rec_norm.strip().lower() != 'fail':
        return (rec_norm, None)
    return (raw_gt, None)

def make_row_key(rec: Dict[str, Any], idx: int) -> str:
    rid = safe_str(rec.get('id'))
    if rec.get('attempt') is not None:
        return f"{rid}::att{safe_int(rec.get('attempt'), 0)}"
    return f'{rid}::idx{idx}'

def load_done_row_keys(output_path: str) -> Set[str]:
    done: Set[str] = set()
    if not output_path or not os.path.exists(output_path):
        return done
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
                rk = obj.get('_row_key')
                if isinstance(rk, str) and rk:
                    done.add(rk)
                    continue
                rid = safe_str(obj.get('id'))
                if not rid:
                    continue
                if obj.get('attempt') is not None:
                    done.add(f"{rid}::att{safe_int(obj.get('attempt'), 0)}")
    except Exception:
        pass
    return done

async def infer_one(solver: AsyncLLMClient, judge_model: Optional[AsyncLLMClient], rec: Dict[str, Any], type_info: Dict[str, Dict[str, Any]], mm_root: Optional[str], row_key: str, no_images: bool) -> Dict[str, Any]:
    rid = safe_str(rec.get('id'))
    question = (rec.get('question') or '').strip()
    modeling = rec.get('modeling')
    if not isinstance(modeling, str) or not modeling.strip():
        raise ValueError("Expected a nonempty 'modeling' field from generate_modeling.")
    modeling = modeling.strip()
    q_type = get_q_type(rec, type_info)
    gt_answer = (rec.get('gt_answer') or rec.get('answer') or '').strip()
    if no_images:
        image_paths_raw, image_paths_resolved = ([], [])
    else:
        image_paths_raw, image_paths_resolved = normalize_image_paths(rec, mm_root)
    user_prompt = SOLVE_WITH_MODELING_PROMPT_TEMPLATE.format(question=question, modeling=modeling)
    content, meta = await solver.chat_generate(user_prompt=user_prompt, image_paths=image_paths_resolved if image_paths_resolved else None)
    if isinstance(meta.get('usage'), dict):
        meta['usage'].pop('completion_tokens_details', None)
        meta['usage'].pop('prompt_tokens_details', None)
    pred_answer = extract_boxed(content) or ''
    patch: Dict[str, Any] = {'_row_key': row_key, 'id': rid, 'attempt': rec.get('attempt'), 'num_attempts': rec.get('num_attempts'), 'subject': rec.get('subject'), 'source': rec.get('source'), 'image_path': image_paths_raw, 'question': question, 'modeling': modeling, 'gt_answer': (rec.get('gt_answer') or rec.get('answer') or '').strip(), 'type': q_type, 'pred_answer': pred_answer, 'solution': content, 'response': content, 'solver_meta': meta}
    if not pred_answer:
        patch.update({'is_correct': 0, 'judge_backend': 'skipped_no_pred'})
        out = {key: value for key, value in rec.items() if key not in ('score', 'accuracy', 'is_corect')}
        out.update(patch)
        return out
    if judge_model is None:
        patch.update({'is_correct': None, 'judge_backend': 'not_evaluated'})
        out = {key: value for key, value in rec.items() if key not in ('score', 'accuracy', 'is_corect')}
        out.update(patch)
        return out
    score = await judge_one(model=judge_model, gt_answer=gt_answer, pred_answer=pred_answer, q_type=q_type)
    patch.update({'is_correct': float(score) if score is not None else None, 'judge_backend': 'judge_one'})
    out = {key: value for key, value in rec.items() if key not in ('score', 'accuracy', 'is_corect')}
    out.update(patch)
    return out

async def run(input_path: str, output_path: str, model_name: str, temperature: float, top_p: float, max_concurrent: int, resume: bool, api_base: str, api_key: str, max_tokens: int, mm_root: Optional[str], gemini_api_key: Optional[str], judge_model_name: str, judge_concurrency: int, type_path: Optional[str], solver_provider: str, solver_image_mode: str, no_images: bool) -> None:
    solver = AsyncLLMClient(model_name=model_name, provider=solver_provider, base_url=api_base, api_key=api_key, temperature=temperature, top_p=top_p, max_tokens=max_tokens, max_concurrent_requests=max_concurrent, mm_root=mm_root, image_mode=solver_image_mode)
    judge_model: Optional[AsyncLLMClient] = None
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        judge_model = AsyncLLMClient(model_name=judge_model_name, provider='gemini', temperature=0.0, top_p=1.0, max_concurrent_requests=judge_concurrency, system_prompt=None, image_mode='data_url')
    type_info = load_type_info(type_path)
    records: List[Dict[str, Any]] = load_records(input_path)
    done_keys: Set[str] = set()
    if resume:
        done_keys = load_done_row_keys(output_path)
    todo: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, rec in enumerate(records):
        rk = make_row_key(rec, idx)
        if rk in done_keys:
            continue
        todo.append((idx, rec, rk))

    async def process(idx: int, rec: Dict[str, Any], rk: str) -> Dict[str, Any]:
        try:
            return await infer_one(solver=solver, judge_model=judge_model, rec=rec, type_info=type_info, mm_root=mm_root, row_key=rk, no_images=no_images)
        except Exception as e:
            rid = safe_str(rec.get('id'))
            out = {key: value for key, value in rec.items() if key not in ('score', 'accuracy', 'is_corect')}
            out.update({'_row_key': rk, 'id': rid, 'attempt': rec.get('attempt'), 'pred_answer': None, 'is_correct': None, 'solution': None, 'response': None, 'solver_meta': None, 'judge_backend': 'error', 'error': f'Solve-with-modeling failed: {type(e).__name__}'})
            return out
    mode = 'a' if resume and os.path.exists(output_path) else 'w'
    with JsonlWriter(output_path, mode=mode, ensure_ascii=False) as writer:
        tasks = [asyncio.create_task(process(idx, rec, rk)) for idx, rec, rk in todo]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc='Solve-with-Modeling'):
            result = await fut
            rk = safe_str(result.get('_row_key'))
            if rk:
                done_keys.add(rk)
            writer.write(result)

def parse_args():
    p = argparse.ArgumentParser(description='Execute physical modeling to solve questions with AsyncLLMClient; optionally score with judge_one.')
    p.add_argument('--input', type=str, required=True, help='Path to input JSON/JSONL (modeling outputs).')
    p.add_argument('--output', type=str, required=True, help='Path to output JSONL.')
    p.add_argument('--model', type=str, required=True, help='Solver model name (vLLM served name or API model).')
    p.add_argument('--api_base', type=str, default=None)
    p.add_argument('--api_key', type=str, default=None)
    p.add_argument('--solver_provider', type=str, default='auto', help='auto|vllm|openai|gemini|compat')
    p.add_argument('--solver_image_mode', type=str, default='auto', help='auto|file_url|data_url')
    p.add_argument('--mm_root', type=str, default='data', help='Dataset/media root for resolving relative image_path.')
    p.add_argument('--max_tokens', type=int, default=10240)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--max_concurrent', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gemini_api_key', type=str, default=None)
    p.add_argument('--judge_model', type=str, default='gemini-2.0-flash')
    p.add_argument('--judge_concurrency', type=int, default=32)
    p.add_argument('--type_file', type=str, default=None, help='Optional JSON/JSONL file providing {"id": ..., "type": ..., "normalized_answer": ...}.')
    p.add_argument('--no_images', action='store_true', help='Do not pass images to the solver even if image_path exists.')
    return p.parse_args()

def main():
    args = parse_args()
    asyncio.run(run(input_path=args.input, output_path=args.output, model_name=args.model, temperature=args.temperature, top_p=args.top_p, max_concurrent=args.max_concurrent, resume=args.resume, api_base=args.api_base, api_key=args.api_key, max_tokens=args.max_tokens, mm_root=args.mm_root, gemini_api_key=args.gemini_api_key if args.gemini_api_key else None, judge_model_name=args.judge_model, judge_concurrency=args.judge_concurrency, type_path=args.type_file, solver_provider=args.solver_provider, solver_image_mode=args.solver_image_mode, no_images=args.no_images))
if __name__ == '__main__':
    main()
