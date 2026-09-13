"""Copy the current engine into novaAAD; never modify its legacy application.

Run only with both checkouts on audio_flo after committing engine changes.
The pin and per-file hashes make the snapshot auditable and reproducible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil


def vendor(destination):
    source = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    for root in (source, destination):
        branch = subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'], text=True).strip()
        if branch != 'audio_flo':
            raise ValueError(f'{root} must be on audio_flo')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'src', 'scripts'], text=True)
    if changed.strip():
        raise ValueError('Commit engine source changes before pinning the vendor snapshot.')
    hashes = {}
    for folder in ('src/nova2026', 'scripts/auditory', 'scripts/attune', 'scripts/getlive'):
        for path in (source/folder).rglob('*.py'):
            relative = path.relative_to(source)
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/'src/nova2026/VENDOR_COMMIT.txt').write_text(commit+'\n')
    (destination/'VENDOR_MANIFEST.json').write_text(json.dumps({'commit': commit, 'sha256': hashes}, indent=2)+'\n')
    (destination/'requirements.txt').write_text(
        'numpy>=2.0\nscipy>=1.14\nmne>=1.12\nmne-lsl>=1.14\nsoxr>=1.1\nantio>=0.5\n'
        'sounddevice>=0.4\npyxdf>=1.16\n# torch is not needed by the AAD import path.\n')
    ignore = destination/'.gitignore'
    if not ignore.exists():
        ignore.write_text('__pycache__/\n*.pyc\n.venv/\nresults/\nrecords/\n')
    tools = destination/'tools'
    tools.mkdir(exist_ok=True)
    for name in ('calibration_session', 'xdf_to_trial'):
        (tools/(name+'.py')).write_text(f'"""Entry point backed by the pinned NOVA implementation."""\n'
            f'from scripts.attune.{name} import main\n\nif __name__ == "__main__":\n    main()\n')
    (destination/'ATTUNE_INTEGRATION.md').write_text(
        '# ATTUNE engine snapshot\n\nThis root-level snapshot is pinned in VENDOR_MANIFEST.json. '
        'The pre-existing neuro-attention-reorganized application is preserved.\n\n'
        'Install requirements.txt; put this checkout and its src directory on PYTHONPATH. '
        'Use python -m scripts.attune.calibration_session and python -m scripts.attune.xdf_to_trial. '
        'Run the attune-ui audio_flo branch with this engine as documented in its docs/NOVA_LIVE.md.\n\n'
        'CPz reference and Fpz ground are operator declarations. Audio onset is taken from the '
        'first DAC callback. Physical residual and diotic decoding accuracy remain unverified.\n')
    print(f'Vendored {len(hashes)} files at {commit}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    vendor(parser.parse_args().destination)
