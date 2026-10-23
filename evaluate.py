"""Evaluate Physical Modeling -> Execution -> judge_one, or score saved predictions."""
import argparse
import asyncio
import json
import math
from pathlib import Path

from phys_utils.io import load_records
from phys_utils.parsing import extract_boxed

ANSWER_TYPES = ('Numerical', 'Symbolic', 'Textual', 'Multi-Choice')
DEFAULT_RTOL = 0.03


class ModelJudgeRequired(RuntimeError):
    """A rule-based check needs an external judge model that was not configured."""


class _UnconfiguredJudge:
    async def chat_generate(self, **kwargs):
        raise ModelJudgeRequired('Model judging is required for this answer; configure --judge_model and its provider.')


def load_type_map(path):
    if not path:
        return {}
    source = Path(path)
    data = load_records(str(source)) if source.suffix == '.jsonl' else json.loads(source.read_text(encoding='utf-8-sig'))
    if isinstance(data, dict):
        items = [(key, value.get('type') if isinstance(value, dict) else value) for key, value in data.items()]
    elif isinstance(data, list) and all(isinstance(row, dict) and row.get('id') is not None for row in data):
        items = [(row['id'], row.get('type')) for row in data]
    else:
        raise ValueError('The type file must contain id/type records or an id-to-type mapping.')
    if any(value not in ANSWER_TYPES for _, value in items):
        raise ValueError('The type file contains an unsupported answer type.')
    return {str(key): value for key, value in items}


def _first(record, names):
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


async def evaluate_records(records, model=None, *, rtol=DEFAULT_RTOL, max_concurrent=16,
                           type_map=None, default_type='Textual'):
    if not records:
        raise ValueError('The input contains no records.')
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError('rtol must be a finite nonnegative number.')
    if max_concurrent < 1:
        raise ValueError('max_concurrent must be positive.')
    if default_type not in ANSWER_TYPES:
        raise ValueError('Unsupported default answer type.')
    type_map = type_map or {}
    prepared = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index}: expected an object.')
        reference = _first(record, ('gt_answer', 'answer', 'GT'))
        if reference is None:
            raise ValueError(f'Record {index}: missing reference answer.')
        prediction = _first(record, ('pred_answer', 'PRED', 'pred'))
        if prediction is None:
            prediction = extract_boxed(_first(record, ('response', 'solution')) or '') or ''
        kind = type_map.get(str(record.get('id')), record.get('type') or default_type)
        if kind not in ANSWER_TYPES:
            raise ValueError(f'Record {index}: unsupported answer type.')
        prepared.append((record, reference, prediction, kind))

    # Import after argument/input validation; --help does not load model dependencies.
    from phys_utils.judge import judge_one

    judge_model = model if model is not None else _UnconfiguredJudge()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item):
        record, reference, prediction, kind = item
        row = dict(record)
        # Remove stale evaluation fields from previous runs before recording this score.
        for name in ('score', 'accuracy', 'is_correct', 'is_corect', 'eval_error', 'rejudge_error', 'rejudge_backend'):
            row.pop(name, None)
        row.update(gt_answer=reference, pred_answer=prediction, type=kind)
        if record.get('error') and not prediction:
            row.update(is_correct=None, judge_backend='generation_error', eval_error='GenerationError')
            return row
        row.pop('error', None)
        if not prediction:
            row.update(is_correct=0, judge_backend='skipped_no_prediction')
            return row
        try:
            async with semaphore:
                score = float(await judge_one(model=judge_model, gt_answer=reference,
                                               pred_answer=prediction, q_type=kind, rtol=rtol))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Judge score must be between zero and one.')
            row.update(is_correct=score, judge_backend='judge_one')
        except Exception as error:
            # Failed judging is not an incorrect answer or a stale successful score.
            row.update(is_correct=None, judge_backend='error',
                       eval_error=type(error).__name__)
        return row

    rows = await asyncio.gather(*(score_one(item) for item in prepared))
    errors = sum(row['is_correct'] is None for row in rows)
    return {'num_records': len(rows), 'num_errors': errors,
            'accuracy': None if errors else sum(row['is_correct'] for row in rows) / len(rows),
            'rtol': rtol, 'records': rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('generate', 'score'), default='generate',
                        help='generate runs modeling, execution, and judging; score evaluates saved predictions.')
    parser.add_argument('--input', required=True, help='Question records, or saved predictions in score mode.')
    parser.add_argument('--output', help='Write a JSON object containing summary metrics and scored records.')
    parser.add_argument('--output_dir', default='outputs/evaluation', help='Directory for generated records and results.json.')
    parser.add_argument('--model', help='Generation model used for both stages unless overridden.')
    parser.add_argument('--provider', default='auto')
    parser.add_argument('--modeling_model')
    parser.add_argument('--solver_model')
    parser.add_argument('--modeling_provider')
    parser.add_argument('--solver_provider')
    parser.add_argument('--modeling_api_base')
    parser.add_argument('--solver_api_base')
    parser.add_argument('--modeling_api_key')
    parser.add_argument('--solver_api_key')
    parser.add_argument('--mm_root', default='data')
    parser.add_argument('--modeling_image_mode', default='auto')
    parser.add_argument('--solver_image_mode', default='auto')
    parser.add_argument('--no_images', action='store_true', help='Do not pass images to the execution stage.')
    parser.add_argument('--num_attempts', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=10240)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL, help='Numerical relative tolerance; default 0.03, matching judge_one.')
    parser.add_argument('--max_concurrent', type=int, default=16)
    parser.add_argument('--type_file', help='Optional external id-to-type JSON/JSONL.')
    parser.add_argument('--default_type', choices=ANSWER_TYPES, default='Textual')
    parser.add_argument('--judge_model', help='Model used for textual judging and rule-based fallback.')
    parser.add_argument('--judge_provider', default='auto', help='auto|vllm|openai|gemini|compat|dashscope')
    parser.add_argument('--api_base', default=None, help='Shared endpoint unless a stage-specific endpoint is set.')
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--judge_api_base', default=None)
    parser.add_argument('--judge_api_key', default=None)
    args = parser.parse_args(argv)
    if args.mode == 'generate' and not ((args.modeling_model or args.model) and (args.solver_model or args.model)):
        parser.error('Generation requires --model, or both --modeling_model and --solver_model.')
    if args.max_concurrent < 1 or args.num_attempts < 1 or args.max_tokens < 1:
        parser.error('Concurrency, attempts, and token limits must be positive.')
    return args


async def generate_predictions(args):
    """Reuse the two existing generation entry points with judging performed once afterward."""
    records = load_records(args.input)
    if not records:
        raise ValueError('The input contains no records.')
    type_map = load_type_map(args.type_file)
    normalized = []
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index}: expected an object.')
        rid, question = record.get('id'), record.get('question')
        if rid is None or not str(rid).strip() or not isinstance(question, str) or not question.strip():
            raise ValueError(f'Record {index}: nonempty id and question are required.')
        try:
            attempt = int(record['attempt']) if record.get('attempt') is not None else None
            attempts = int(record['num_attempts']) if record.get('num_attempts') is not None else args.num_attempts
        except (ValueError, TypeError):
            raise ValueError(f'Record {index}: attempt counts must be positive integers.') from None
        if (attempt is not None and attempt < 1) or attempts < 1:
            raise ValueError(f'Record {index}: attempt counts must be positive integers.')
        key = (str(rid), attempt)
        if key in seen:
            raise ValueError(f'Record {index}: duplicate id/attempt pair.')
        seen.add(key)
        reference = _first(record, ('gt_answer', 'answer', 'GT'))
        if reference is None:
            raise ValueError(f'Record {index}: reference answer is required.')
        kind = type_map.get(str(rid), record.get('type') or args.default_type)
        if kind not in ANSWER_TYPES:
            raise ValueError(f'Record {index}: unsupported answer type.')
        item = dict(record)
        for name in ('score', 'accuracy', 'is_correct', 'is_corect', 'pred_answer', 'PRED', 'pred',
                     'modeling', 'solution', 'response', 'error', 'eval_error', 'judge_backend',
                     'rejudge_error', 'rejudge_backend', 'modeling_meta', 'solver_meta'):
            item.pop(name, None)
        item.update(id=str(rid), question=question.strip(), answer=reference, gt_answer=reference, type=kind)
        item['num_attempts'] = attempts
        if attempt is not None:
            item['attempt'] = attempt
        normalized.append(item)
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / 'inputs.jsonl'
    modeling_path = directory / 'modeling.jsonl'
    solutions_path = directory / 'solutions.jsonl'
    reserved = (input_path, modeling_path, solutions_path)
    output_path = Path(args.output).resolve() if args.output else directory / 'results.json'
    if Path(args.input).resolve() in reserved or output_path in reserved or output_path == Path(args.input).resolve():
        raise ValueError('Input and result paths must be distinct from generation artifacts.')
    from phys_utils.io import JsonlWriter
    from scripts import generate_modeling, solve_with_modeling
    with JsonlWriter(str(input_path)) as writer:
        for record in normalized:
            writer.write(record)
    common = dict(temperature=args.temperature, top_p=args.top_p, max_concurrent=args.max_concurrent,
                  resume=False, max_tokens=args.max_tokens, mm_root=args.mm_root, type_path=args.type_file)
    await generate_modeling.run(input_path=str(input_path), output_path=str(modeling_path),
        model_name=args.modeling_model or args.model, api_base=args.modeling_api_base or args.api_base,
        api_key=args.modeling_api_key or args.api_key, modeling_provider=args.modeling_provider or args.provider,
        modeling_image_mode=args.modeling_image_mode, modeling_prompt_path=None,
        num_attempts=args.num_attempts, **common)
    await solve_with_modeling.run(input_path=str(modeling_path), output_path=str(solutions_path),
        model_name=args.solver_model or args.model, api_base=args.solver_api_base or args.api_base,
        api_key=args.solver_api_key or args.api_key, solver_provider=args.solver_provider or args.provider,
        solver_image_mode=args.solver_image_mode, no_images=args.no_images, gemini_api_key=None,
        judge_model_name='', judge_concurrency=args.max_concurrent, **common)
    generated = load_records(str(solutions_path))
    expected = sum(1 if item.get('attempt') is not None else item['num_attempts'] for item in normalized)
    if len(generated) != expected:
        raise RuntimeError('Generation record count does not match the requested attempts.')
    return generated


async def run(args):
    if not math.isfinite(args.rtol) or args.rtol < 0:
        raise ValueError('rtol must be a finite nonnegative number.')
    records = await generate_predictions(args) if args.mode == 'generate' else load_records(args.input)
    model = None
    if args.judge_model:
        from phys_utils.llm_client import AsyncLLMClient
        model = AsyncLLMClient(model_name=args.judge_model, provider=args.judge_provider,
                               base_url=args.judge_api_base or args.api_base,
                               api_key=args.judge_api_key or args.api_key, temperature=0.0,
                               max_concurrent_requests=args.max_concurrent)
    result = await evaluate_records(records, model=model, rtol=args.rtol,
                                    max_concurrent=args.max_concurrent, type_map=load_type_map(args.type_file),
                                    default_type=args.default_type)
    result['judge_model'] = args.judge_model
    result['mode'] = args.mode
    text = json.dumps(result, ensure_ascii=False, indent=2)
    destination = Path(args.output) if args.output else Path(args.output_dir) / 'results.json'
    if destination.resolve() == Path(args.input).resolve():
        raise ValueError('The evaluation output must not overwrite the input file.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 1 if result['num_errors'] else 0


def main():
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == '__main__':
    main()
