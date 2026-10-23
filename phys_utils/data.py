"""Dataset sources, normalization, media export, and deterministic sampling."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator
from urllib.parse import quote


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    patterns: tuple[str, ...]
    files: tuple[str, ...]
    role: str


SOURCES = {
    'physreason': Source('zhibei1204/PhysReason', 'b63b7fa6e6bc99563038cafb1e136b060da2f91d',
                         ('PhysReason-full.zip',), ('PhysReason-full.zip',), 'multimodal'),
    'phyx': Source('Cloudriver/PhyX', '5a23a2762d0626b5f319bd725138cc9b18b45d8f',
                   ('data_llms_eval/PhyX_OE.parquet',), ('data_llms_eval/PhyX_OE.parquet',), 'multimodal'),
    'seephys': Source('SeePhys/SeePhys', '9e98ec058e44f65a9f6fc84de60b0f3f05c4d124',
                      ('data/train-*.parquet',), ('data/train-*.parquet',), 'multimodal'),
    'scp116k': Source('EricLu/SCP-116K', '99acd159e5dcc65c7679def10231b75281989fc1',
                      ('SCP_116K_in_open_source_format.jsonl',), ('SCP_116K_in_open_source_format.jsonl',), 'text'),
    'megascience': Source('MegaScience/MegaScience', '8df5586005374acba25aecc4f5469ce30fec605c',
                          ('data/train-*.parquet',), ('data/train-*.parquet',), 'text'),
    'physunibench': Source('PrismaX/PhysUniBench', '726c38afd9038d4d568d89adc3b744e08898a2f0',
                           ('PhysUnivBench_en_MCQ.json', 'images/*'), ('PhysUnivBench_en_MCQ.json',), 'evaluation'),
    'mmk12': Source('FanqingM/MMK12', '372a609268ea79b5e78d90ab173e02c37b486163',
                    ('data/test-*.parquet',), ('data/test-*.parquet',), 'evaluation'),
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}:{value}'.encode()).digest(), 'big')


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes its dataset directory.')
    return path


def extract_archive(path: Path, destination: Path) -> None:
    """Extract only regular entries below destination; reject traversal and links."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 20000 or sum(item.file_size for item in entries) > 2 * 1024**3:
            raise ValueError('Archive exceeds the extraction limits.')
        for item in entries:
            name = item.filename.replace('\\', '/')
            if re.match(r'^[A-Za-z]:', name) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains an unsupported entry.')
            inside(destination, name)
        for item in entries:
            if item.is_dir():
                continue
            target = inside(destination, item.filename.replace('\\', '/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def iter_rows(path: Path) -> Iterator[dict]:
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Expected JSONL objects.')
                    yield row
    elif path.suffix == '.json':
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(value, dict) and isinstance(value.get('records'), list):
            value = value['records']
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError('Expected a JSON array of records.')
        yield from value
    elif path.suffix == '.parquet':
        import pyarrow.parquet as parquet
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        raise ValueError('Unsupported record file extension.')


class MediaStore:
    def __init__(self, output: Path):
        self.output = output

    def save(self, value, source_root: Path) -> str:
        from PIL import Image
        if isinstance(value, dict):
            raw = value.get('bytes')
            value = raw if raw is not None else value.get('path')
        if isinstance(value, str):
            value = inside(source_root, value).read_bytes()
        if not isinstance(value, (bytes, bytearray)) or not value:
            raise ValueError('Invalid image payload.')
        raw = bytes(value)
        with Image.open(BytesIO(raw)) as image:
            extension = (image.format or '').lower()
            image.verify()
        extension = {'jpeg': 'jpg'}.get(extension, extension)
        if extension not in ('png', 'jpg', 'webp', 'gif', 'bmp', 'tiff'):
            raise ValueError('Unsupported image format.')
        relative = f'images/{hashlib.sha256(raw).hexdigest()}.{extension}'
        destination = self.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(raw)
        return relative


def text(value) -> str:
    return str(value).strip() if value is not None else ''


def first(row: dict, *keys: str) -> str:
    return next((text(row.get(key)) for key in keys if text(row.get(key))), '')


def base_record(source: str, source_id, row: dict, revision: str) -> dict:
    original_id = text(source_id)
    identifier = f'{source}:{quote(original_id, safe="")}'
    return {'id': identifier, 'parent_id': identifier, 'source_id': original_id,
            'source': source, 'source_revision': revision, 'subject': 'physics',
            'question': '', 'gt_answer': '', 'reference_solution': '',
            'image_path': [], 'type': 'Textual',
            'language': first(row, 'language', 'lang').lower()}


def normalize_row(source: str, row: dict, index: int, root: Path,
                  media: MediaStore, revision: str) -> dict:
    source_id = next((row[key] for key in ('id', 'index', 'problem_id') if row.get(key) is not None), index)
    record = base_record(source, source_id, row, revision)
    if source == 'scp116k':
        record.update(subject=first(row, 'domain').lower(), question=first(row, 'problem'),
                      gt_answer=first(row, 'extract_solution'),
                      reference_solution=first(row, 'extract_solution', 'r1_response'))
    elif source == 'megascience':
        record.update(subject=first(row, 'subject').lower(), question=first(row, 'question'),
                      gt_answer=first(row, 'reference_answer', 'answer'),
                      reference_solution=first(row, 'answer'))
    else:
        question = first(row, 'question', 'problem')
        description = first(row, 'question_description_simplified', 'question_description')
        if description and description not in question:
            question = description + '\n' + question
        options = row.get('options')
        if options:
            option_text = options if isinstance(options, str) else '\n'.join(map(str, options))
            if option_text not in question:
                question += '\n' + option_text
        record.update(question=question, gt_answer=first(row, 'answer', 'gt_answer'),
                      reference_solution=first(row, 'reasoning', 'parsing', 'solution'))
        if source == 'mmk12':
            record['subject'] = first(row, 'subject').lower()
        if source in ('physunibench', 'mmk12'):
            record['type'] = 'Multi-Choice'
        images = row.get('images')
        if images is None:
            images = [row['image']] if row.get('image') is not None else []
        if not isinstance(images, list):
            raise ValueError('Expected an image list.')
        image_root = root / 'images' if source == 'physunibench' else root
        record['image_path'] = [media.save(value, image_root) for value in images if value is not None]
    kind = row.get('type')
    if kind in ('Numerical', 'Symbolic', 'Textual', 'Multi-Choice'):
        record['type'] = kind
    return record


def physreason_records(path: Path, media: MediaStore, revision: str) -> Iterator[dict]:
    row = json.loads(path.read_text(encoding='utf-8-sig'))
    structure = row.get('question_structure', {})
    keys = sorted((key for key in structure if key.startswith('sub_question_')),
                  key=lambda key: int(key.rsplit('_', 1)[1]))
    if not keys:
        raise ValueError('No PhysReason sub-questions.')
    answers = row.get('answer', [])
    if isinstance(answers, list) and len(answers) != len(keys):
        raise ValueError('PhysReason sub-question/answer counts differ.')
    images = [media.save(value, path.parent) for value in row.get('question_image_list', [])]
    for index, key in enumerate(keys):
        record = base_record('physreason', path.parent.name, row, revision)
        record['id'] += '/' + key
        context = text(structure.get('context'))
        parts = [f'({i + 1}) {text(structure[name])}' for i, name in enumerate(keys[:index + 1])]
        record['question'] = '\n'.join([context, *parts, f'Solve sub-question ({index + 1}).']).strip()
        record['gt_answer'] = text(answers.get(key)) if isinstance(answers, dict) else text(answers[index])
        steps = row.get('explanation_steps', {}).get(key, {})
        record['reference_solution'] = '\n'.join(map(str, steps.values())) if isinstance(steps, dict) else text(steps)
        record['image_path'] = images
        yield record


def source_records(directory: Path, source: str, media: MediaStore) -> Iterator[dict]:
    manifest = json.loads((directory / source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('dataset') != source or manifest.get('repo_id') != SOURCES[source].repo:
        raise ValueError('Dataset manifest does not match the selected source.')
    revision = manifest['revision']
    root = inside(directory / source, revision)
    entries = manifest.get('files', [])
    for item in entries:
        path = inside(root, item['path'])
        if digest_file(path) != item['sha256']:
            raise ValueError('Downloaded file checksum does not match its manifest.')
    if source == 'physreason':
        extracted = root / 'extracted'
        extract_archive(root / 'PhysReason-full.zip', extracted)
        for path in sorted(extracted.rglob('problem.json')):
            try:
                yield from physreason_records(path, media, revision)
            except (ValueError, OSError, KeyError, TypeError):
                yield {'source': source, 'id': f'{source}:{path.parent.name}', '_error': 'invalid_source_record'}
        return
    index = 0
    for pattern in SOURCES[source].files:
        for path in sorted(root.glob(pattern)):
            for row in iter_rows(path):
                try:
                    # Filter non-physics text/MCQ rows before decoding image payloads.
                    subject = first(row, 'domain' if source == 'scp116k' else 'subject').lower()
                    if source in ('scp116k', 'megascience', 'mmk12') and subject not in ('physics', '物理'):
                        yield {'source': source, 'id': f'{source}:row-{index}', '_error': 'non_physics'}
                    else:
                        yield normalize_row(source, row, index, root, media, revision)
                except (ValueError, OSError, KeyError, TypeError):
                    yield {'source': source, 'id': f'{source}:row-{index}', '_error': 'invalid_source_record'}
                index += 1
    if not index:
        raise ValueError('No source records found in the downloaded files.')


def english(record: dict) -> bool:
    language = record.get('language', '').lower()
    if language:
        return language in ('en', 'eng', 'english')
    import langid
    question = re.sub(r'\\[A-Za-z]+|\$[^$]*\$', ' ', record['question'])
    return langid.classify(question)[0] == 'en'


def content_key(record: dict) -> str:
    question = ' '.join(record['question'].casefold().split())
    images = sorted(Path(path).stem for path in record.get('image_path', []))
    return hashlib.sha256(json.dumps([question, images], ensure_ascii=False).encode()).hexdigest()
