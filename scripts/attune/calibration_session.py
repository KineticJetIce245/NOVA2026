"""Play cued calibration; record EEG and AAD_Markers with LabRecorder to XDF.

The audio:onset marker uses the first callback's DAC timestamp mapped to LSL,
never the device-open or session:start time. Physical residual is unverified.
"""
import argparse
import numpy as np
from nova2026.auditory.audio import DichoticMixer, DioticMixer
from nova2026.auditory.config import AuditoryConfig
from .events import EpochLedger, Markers
from .stimuli import load_candidates, check_candidates


def calibration_callback(audio, rate, presentation, block_seconds, markers, clock,
                         *, trial_id='calibration', callback_stop=StopIteration):
    if presentation not in ('dichotic', 'diotic') or not np.isfinite(block_seconds) or block_seconds < 10:
        raise ValueError('Select an arm and calibration blocks of at least 10 seconds.')
    mixer = (DichoticMixer if presentation == 'dichotic' else DioticMixer)(rate)
    position = 0
    epoch = None
    last_block = -1
    ledger = EpochLedger()

    def callback(outdata, count, time_info, status):
        nonlocal position, epoch, last_block
        if status:
            raise RuntimeError(f'Audio status during calibration: {status}')
        audible = clock()+time_info.outputBufferDacTime-time_info.currentTime
        if epoch is None:
            epoch = audible
            markers.push({'event': 'audio:onset', 'presentation': presentation,
                          'trial_id': trial_id, 'acoustic_residual': None}, audible)
        outdata[:] = 0
        if position >= len(audio):
            markers.push({'event': 'audio:end'}, audible)
            raise callback_stop
        block = int(position/rate/block_seconds)
        if block != last_block:
            start = epoch+block*block_seconds
            end = min(epoch+(block+1)*block_seconds, epoch+len(audio)/rate)
            event = ledger.add(start, end, trial_id, 'A' if block % 2 == 0 else 'B',
                               presentation, str(block))
            markers.push(event, start)
            print(f'Attend {event.cued_side} ({"left" if block % 2 == 0 else "right"})' if presentation == 'dichotic'
                  else f'Attend talker {event.cued_side}', flush=True)
            last_block = block
        chunk = audio[position:position+count]
        outdata[:len(chunk)] = mixer.process(chunk, [1, 1])
        position += len(chunk)
    return callback


def main():
    import sounddevice as sd
    from mne_lsl.lsl import local_clock
    from threading import Event
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--form', choices=('pair', 'dichotic'), required=True)
    p.add_argument('--audio', nargs='+', required=True)
    p.add_argument('--presentation', choices=('dichotic', 'diotic'), required=True)
    p.add_argument('--block-seconds', type=float, default=60.)
    p.add_argument('--device', required=True)
    p.add_argument('--trial-id', required=True)
    args = p.parse_args()
    audio, rate = load_candidates(args.form, *args.audio)
    print('Candidate envelope correlation:', check_candidates(audio, rate, AuditoryConfig()))
    markers = Markers()
    finished = Event()
    callback = calibration_callback(audio, rate, args.presentation, args.block_seconds,
                                    markers, local_clock, trial_id=args.trial_id,
                                    callback_stop=sd.CallbackStop)
    with sd.OutputStream(samplerate=rate, channels=2, dtype='float32', device=args.device,
                         blocksize=round(rate*.032), callback=callback, finished_callback=finished.set):
        finished.wait(len(audio)/rate+5)


if __name__ == '__main__':
    main()
