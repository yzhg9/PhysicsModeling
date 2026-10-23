"""Prepare source records or validate generated physical modeling trajectories."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path

from phys_utils.data import (SOURCES, MediaStore, content_key, english, source_records,
                             stable_rank, write_json)
from phys_utils.io import JsonlWriter, load_records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('source', 'modeling'), default='source')
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--datasets', nargs='+', choices=tuple(SOURCES))
    parser.add_argument('--input', type=Path, help='Generated modeling records for the modeling stage.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--language', choices=('en', 'any'), default='en')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--text-limit', type=int, default=12000, help='Combined text sample cap; zero keeps all.')
    parser.add_argument('--model', help='Quality evaluation model.')
    parser.add_argument('--solver_model')
    parser.add_argument('--judge_model')
    parser.add_argument('--provider', default='auto')
    parser.add_argument('--api_base')
    parser.add_argument('--api_key')
    for role in ('solver', 'judge'):
        parser.add_argument(f'--{role}_provider')
        parser.add_argument(f'--{role}_api_base')
        parser.add_argument(f'--{role}_api_key')
    parser.add_argument('--mm_root', type=Path)
    parser.add_argument('--max_concurrent', type=int, default=8)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--keep-all-attempts', action='store_true')
    args = parser.parse_args(argv)
    if args.text_limit < 0:
        parser.error('--text-limit must be nonnegative.')
    if args.max_concurrent < 1 or args.max_tokens < 1:
        parser.error('Concurrency and token limits must be positive.')
    if args.stage == 'modeling' and (not args.input or not all((args.model, args.solver_model, args.judge_model))):
        parser.error('modeling requires --input, --model, --solver_model, and --judge_model.')
    source = args.input if args.stage == 'modeling' else args.input_dir
    output = args.output_dir.resolve()
    if source and (output == source.resolve() or source.resolve().is_relative_to(output)
                   or output.is_relative_to(source.resolve())):
        parser.error('Input and output locations must be separate.')
    return args


def prepare_sources(args):
    names = args.datasets or [name for name in SOURCES if (args.input_dir / name / 'manifest.json').is_file()]
    if not names:
        raise ValueError('No dataset manifests found; run scripts.download_data first.')
    media = MediaStore(args.output_dir)
    counts = Counter()
    reasons = Counter()
    records, text_heap = [], []
    serial = 0
    with JsonlWriter(str(args.output_dir / 'decisions.jsonl')) as decisions:
        def reject(record, reason):
            reasons[reason] += 1
            decisions.write({'id': record.get('id'), 'source': record.get('source'), 'reason': reason})

        for name in names:
            for record in source_records(args.input_dir, name, media):
                counts[name] += 1
                reason = record.get('_error')
                if not reason and not record['question'].strip():
                    reason = 'empty_question'
                if not reason and not record['gt_answer'].strip():
                    reason = 'empty_reference'
                if not reason and args.language == 'en' and not english(record):
                    reason = 'non_english'
                if reason:
                    reject(record, reason)
                    continue
                if SOURCES[name].role == 'text' and args.text_limit:
                    serial += 1
                    rank = stable_rank(record['id'], args.seed)
                    entry = (-rank, serial, record)
                    if len(text_heap) < args.text_limit:
                        heapq.heappush(text_heap, entry)
                    else:
                        removed = heapq.heappushpop(text_heap, entry)
                        reject(removed[2], 'text_sample_limit')
                else:
                    records.append(record)
        records.extend(entry[2] for entry in text_heap)

        seen = set()
        outputs = {key: JsonlWriter(str(args.output_dir / filename)) for key, filename in
                   [('text', 'text_pool.jsonl'), ('multimodal', 'multimodal_pool.jsonl'),
                    ('evaluation', 'evaluation.jsonl')]}
        kept = Counter()
        try:
            records.sort(key=lambda row: (SOURCES[row['source']].role != 'evaluation', row['id']))
            identifiers = set()
            for record in records:
                if record['id'] in identifiers:
                    raise ValueError('Duplicate source identifier after normalization.')
                identifiers.add(record['id'])
                key = content_key(record)
                if key in seen:
                    reject(record, 'duplicate_question')
                    continue
                seen.add(key)
                role = SOURCES[record['source']].role
                outputs[role].write(record)
                kept[role] += 1
        finally:
            for writer in outputs.values():
                writer.close()
    if not sum(kept.values()):
        raise ValueError('No records passed source filtering.')
    report = {'stage': 'source', 'num_input': sum(counts.values()), 'input_by_source': dict(counts),
              'kept': dict(kept), 'rejected': dict(reasons), 'seed': args.seed,
              'text_limit': args.text_limit,
              'language': args.language}
    write_json(args.output_dir / 'summary.json', report)
    return report


QUALITY_FIELDS = ('internal_coherence', 'multimodal_consistency', 'modeling_specificity', 'physical_consistency')


def parse_quality(value: str) -> dict:
    value = value.strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or any(type(parsed.get(key)) is not bool for key in QUALITY_FIELDS):
        raise ValueError('Quality evaluation must return the four boolean criteria.')
    return {key: parsed[key] for key in QUALITY_FIELDS}


def sft_record(record: dict, media_root: Path | None) -> dict:
    images = record.get('image_path') or []
    resolved = []
    for value in images:
        path = Path(value)
        if not path.is_absolute():
            if media_root is None:
                raise ValueError('Relative media paths require --mm_root.')
            path = media_root / path
        if not path.is_file():
            raise ValueError('SFT image does not exist.')
        resolved.append(str(path.resolve()))
    modeling = record['modeling'].strip()
    if not modeling.startswith('<modeling>'):
        modeling = '<modeling>' + modeling + '</modeling>'
    result = {'instruction': '<image>' * len(resolved) + record['question'], 'input': '',
              'output': modeling + '\n' + record['solution']}
    if resolved:
        result['images'] = resolved
    return result


async def validate_modeling(args, *, quality_client=None, solver=None, judge=None,
                            execute=None, quality_template=None):
    from phys_utils.prompt_loader import load_prompt
    if quality_template is None:
        quality_template = load_prompt('scripts.filter_data.MODELING_QUALITY_PROMPT')
    if execute is None:
        from scripts.solve_with_modeling import infer_one
        execute = infer_one
    if any(client is None for client in (quality_client, solver, judge)):
        from phys_utils.llm_client import AsyncLLMClient

        def client(model, role=''):
            prefix = role + '_' if role else ''
            return AsyncLLMClient(model_name=model,
                provider=(getattr(args, prefix + 'provider', None) or args.provider),
                base_url=(getattr(args, prefix + 'api_base', None) or args.api_base),
                api_key=(getattr(args, prefix + 'api_key', None) or args.api_key),
                temperature=0, top_p=1, max_tokens=args.max_tokens,
                max_concurrent_requests=args.max_concurrent,
                mm_root=str(args.mm_root) if args.mm_root else None)
        quality_client = quality_client or client(args.model)
        solver = solver or client(args.solver_model, 'solver')
        judge = judge or client(args.judge_model, 'judge')

    records = load_records(str(args.input))
    if not records:
        raise ValueError('No modeling records supplied.')
    groups = defaultdict(list)
    keys = set()
    for record in records:
        if not isinstance(record, dict) or not record.get('id'):
            raise ValueError('Each modeling record requires an id.')
        attempt = int(record.get('attempt') or 0)
        key = (str(record['id']), attempt)
        if key in keys:
            raise ValueError('Duplicate id/attempt pair.')
        keys.add(key)
        groups[key[0]].append((attempt, record))
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def group_task(group):
        accepted, decisions = [], []
        async with semaphore:
            for attempt, record in sorted(group, key=lambda item: item[0]):
                decision = {'id': record['id'], 'attempt': attempt}
                if accepted and not args.keep_all_attempts:
                    decisions.append(dict(decision, reason='another_attempt_selected'))
                    continue
                if not all(isinstance(record.get(key), str) and record[key].strip()
                           for key in ('question', 'modeling', 'gt_answer')):
                    decisions.append(dict(decision, reason='invalid_modeling_record'))
                    continue
                try:
                    images = []
                    for path in record.get('image_path') or []:
                        media = Path(path)
                        if not media.is_absolute():
                            if args.mm_root is None:
                                raise ValueError('Relative media paths require --mm_root.')
                            media = args.mm_root / media
                        if not media.is_file():
                            raise ValueError('Image does not exist.')
                        images.append(str(media.resolve()))
                    response, _ = await quality_client.chat_generate(
                        user_prompt=quality_template.format(question=record['question'], modeling=record['modeling']),
                        image_paths=images or None)
                    criteria = parse_quality(response)
                    decision['quality'] = criteria
                    if not all(criteria.values()):
                        decisions.append(dict(decision, reason='modeling_quality'))
                        continue
                    result = await execute(solver=solver, judge_model=judge, rec=record,
                        type_info={}, mm_root=str(args.mm_root) if args.mm_root else None,
                        row_key=f'{record["id"]}::att{attempt}', no_images=False)
                    score = result.get('is_correct')
                    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                        raise ValueError('Execution judging did not produce a valid correctness value.')
                    decision['is_correct'] = score
                    if score != 1:
                        decisions.append(dict(decision, reason='incorrect_execution'))
                        continue
                    result['modeling_quality'] = criteria
                    training_record = sft_record(result, args.mm_root)
                    accepted.append((result, training_record))
                    decisions.append(dict(decision, reason='accepted'))
                except Exception as error:
                    decisions.append(dict(decision, reason='error', error=type(error).__name__))
        return accepted, decisions

    results = await asyncio.gather(*(group_task(groups[key]) for key in sorted(groups)))
    counts = Counter()
    with JsonlWriter(str(args.output_dir / 'accepted.jsonl')) as accepted_output, \
         JsonlWriter(str(args.output_dir / 'sft.jsonl')) as training_output, \
         JsonlWriter(str(args.output_dir / 'decisions.jsonl')) as decision_output:
        for accepted, decisions in results:
            for record, training_record in accepted:
                accepted_output.write(record)
                training_output.write(training_record)
            for decision in decisions:
                decision_output.write(decision)
                counts[decision['reason']] += 1
    report = {'stage': 'modeling', 'num_input': len(records), 'num_accepted': counts['accepted'],
              'num_errors': counts['error'], 'decisions': dict(counts)}
    write_json(args.output_dir / 'summary.json', report)
    return report


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = prepare_sources(args) if args.stage == 'source' else asyncio.run(validate_modeling(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get('num_errors'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
