"""Finite, separate-process PlayerLSL source for recorded-EEG software demos."""
import argparse
from pathlib import Path
import time


def run(raw_path, name, ready=None):
    from mne_lsl.player import PlayerLSL
    player = PlayerLSL(raw_path, chunk_size=16, name=name, n_repeat=1)
    player.start()
    if ready:
        Path(ready).write_text('ready')
    try:
        while player.running:
            time.sleep(.1)
    finally:
        if player.running:
            player.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--ready')
    args = parser.parse_args()
    run(args.raw, args.name, args.ready)
