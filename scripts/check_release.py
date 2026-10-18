"""Check source/package hygiene without loading model weights or contacting services."""
import ast
import argparse
import json
from pathlib import Path
import re
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml',
            'src/awm/cli.py', 'src/awm/configs/inference.yaml',
            '.github/workflows/ci.yml', 'docs/TRAINING.md', 'docs/DATA.md',
            'docs/INFERENCE.md', 'docs/checkpoint-download-manifest.json',
            'scripts/download_checkpoint.py', 'src/awm/training/runner.py',
            'src/awm/configs/training.yaml']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    errors = [f'Missing {name}' for name in REQUIRED if not (ROOT / name).is_file()]
    package = ROOT / 'src/awm'
    for path in package.rglob('*.py'):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = list(path.relative_to(ROOT / 'src').with_suffix('').parts[:-1])
                    if node.level > 1:
                        relative = relative[:-(node.level - 1)]
                    modules = ['.'.join(relative + (node.module or '').split('.')).rstrip('.')]
                elif node.module:
                    modules = [node.module]
            for module in modules:
                if module == 'awm' or module.startswith('awm.'):
                    target = (ROOT / 'src').joinpath(*module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        errors.append(f'{path.relative_to(ROOT)}: missing internal module {module}')
        if path.stat().st_size > 5_000_000:
            errors.append(f'Unexpected large source file {path.relative_to(ROOT)}')
        if re.search(r'/lpai/|/mnt/volumes/|10\.196\.\d+\.\d+|/Users/yangyihui', path.read_text()):
            errors.append(f'Machine-specific reference in {path.relative_to(ROOT)}')
    wheels = list(args.dist_dir.glob('*.whl'))
    sources = list(args.dist_dir.glob('*.tar.gz'))
    if not wheels or not sources:
        errors.append('Build both wheel and source distribution before checking the release')
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            for required in ['awm/cli.py', 'awm/configs/inference.yaml', 'awm/configs/checkpoint_sha256.json', 'awm/configs/training.yaml',
                             'awm/training/runner.py', 'awm/training/objective.py',
                             'awm/scripts/precompute_nav_wam_latents.py', 'awm/scripts/precompute_nav_t2v_latents.py']:
                if required not in names:
                    errors.append(f'{wheel.name}: missing {required}')
            for name in names:
                if name.endswith(('.pt', '.pth', '.safetensors')) or '/checkpoints/' in name:
                    errors.append(f'{wheel.name}: bundled weights {name}')
                if archive.getinfo(name).file_size > 5_000_000:
                    errors.append(f'{wheel.name}: unexpected large file {name}')
    for source in sources:
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if (member.issym() or member.islnk() or member.size > 5_000_000
                        or any(part in {'checkpoints', 'outputs', '.git', '.env'} for part in parts)
                        or member.name.endswith(('.pt', '.pth', '.safetensors'))):
                    errors.append(f'{source.name}: unexpected artifact {member.name}')
    print(json.dumps({'source_package_checks': 'failed' if errors else 'passed', 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
