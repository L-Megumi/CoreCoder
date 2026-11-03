"""Offline installed-package smoke checks; run outside the source checkout."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import corecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reject-source-root', type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    module = Path(corecoder.__file__).resolve()
    if args.reject_source_root and module.is_relative_to(args.reject_source_root.resolve()):
        raise RuntimeError('Smoke check imported source checkout instead of installed wheel')
    assert corecoder.__version__ == importlib.metadata.version('corecoder')
    env = {k: v for k, v in os.environ.items() if not k.endswith('_API_KEY') and k != 'PYTHONPATH'}
    commands = [
        ('help', [sys.executable, '-m', 'corecoder.researchcoder', '--help']),
        ('workflow', [sys.executable, '-m', 'corecoder.researchcoder', 'demo', '--output-dir', str(output / 'workflow')]),
        ('symbol', [sys.executable, '-m', 'corecoder.symbol_demo', '--output-dir', str(output / 'symbol')]),
        ('feedback', [sys.executable, '-m', 'corecoder.feedback_demo', '--output-dir', str(output / 'feedback')]),
        ('coding', [sys.executable, '-m', 'corecoder', '--demo']),
    ]
    rows = []
    for name, command in commands:
        result = subprocess.run(command, cwd=output, env=env, capture_output=True, text=True, timeout=90, check=False)
        (output / (name + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
        rows.append({'name': name, 'exit_code': result.returncode})
    summary = {'version': corecoder.__version__, 'installed_module': str(module), 'checks': rows,
               'passed': all(row['exit_code'] == 0 for row in rows), 'provider_requests': 0,
               'scope': 'Scripted model decisions, actual file edits and verification commands'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    return 0 if summary['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
