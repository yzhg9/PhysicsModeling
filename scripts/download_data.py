"""Download versioned source datasets and record file checksums."""
import argparse
import fnmatch
from pathlib import Path

from phys_utils.data import SOURCES, digest_file, inside, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=(*SOURCES, 'all'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--revision', action='append', default=[], metavar='DATASET=REVISION')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true', help='List supported sources without downloading.')
    args = parser.parse_args(argv)
    if not args.list and not args.datasets:
        parser.error('--datasets is required unless --list is used.')
    if args.max_workers < 1:
        parser.error('--max-workers must be positive.')
    revisions = {}
    for value in args.revision:
        name, separator, revision = value.partition('=')
        if not separator or name not in SOURCES or not revision:
            parser.error('--revision must be DATASET=REVISION for a supported source.')
        revisions[name] = revision
    args.revisions = revisions
    return args


def run(args):
    if args.list:
        for name, source in SOURCES.items():
            print(f'{name}\t{source.repo}\t{source.role}')
        return
    from huggingface_hub import HfApi, snapshot_download
    names = list(SOURCES) if 'all' in args.datasets else list(dict.fromkeys(args.datasets))
    api = HfApi()
    for name in names:
        source = SOURCES[name]
        info = api.dataset_info(source.repo, revision=args.revisions.get(name, source.revision))
        revision = info.sha
        if not revision or len(revision) != 40:
            raise ValueError('Dataset revision did not resolve to a commit hash.')
        selected = sorted(item.rfilename for item in info.siblings
                          if any(fnmatch.fnmatchcase(item.rfilename, pattern) for pattern in source.patterns))
        if not selected or any(not any(fnmatch.fnmatchcase(path, pattern) for path in selected)
                               for pattern in source.files):
            raise ValueError(f'{name}: expected source files were not found.')
        print(f'{name}: {source.repo}@{revision}, {len(selected)} files')
        if args.dry_run:
            continue
        directory = args.output_dir / name
        target = inside(directory, revision)
        snapshot_download(repo_id=source.repo, repo_type='dataset', revision=revision,
                          local_dir=target, allow_patterns=selected, max_workers=args.max_workers)
        entries = []
        for relative in selected:
            path = inside(target, relative)
            entries.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest_file(path)})
        write_json(directory / 'manifest.json', {'dataset': name, 'repo_id': source.repo,
                   'revision': revision, 'files': entries})


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
