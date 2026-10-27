"""
Record custom keyword samples for KWS training.

Run locally on your laptop (needs a mic + `pip install sounddevice scipy numpy`).
Produces 1-second, 16kHz, mono, 16-bit WAV files matching Speech Commands format.

Usage:
    python record_samples.py --label "marvin" --count 50 --outdir data/keyword

Tips for good data (this matters more than model tweaks):
  - Vary distance: some samples close (30cm), some far (2-3m)
  - Vary tone: normal, quiet, loud, questioning
  - Vary background: silence, TV/music low, fan noise, another room
  - Get 2-3 different speakers if possible (teammates) - improves generalization a lot
  - Say the word naturally, don't over-enunciate
"""
import argparse
import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 1.0  # seconds


def record_one(path):
    print("  Recording in: 3", end="\r")
    time.sleep(0.4)
    print("  Recording in: 2", end="\r")
    time.sleep(0.4)
    print("  Recording in: 1", end="\r")
    time.sleep(0.4)
    print("  Recording NOW - speak!     ")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    write(path, SAMPLE_RATE, audio)
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="keyword label, e.g. 'Sonic'")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--start_index", type=int, default=0, help="resume from this index")
    args = ap.parse_args()

    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)

    print(f"Recording {args.count} samples of '{args.label}' into {outdir}/")
    print("Press Ctrl+C anytime to stop early - already-saved files are kept.\n")

    for i in range(args.start_index, args.start_index + args.count):
        path = os.path.join(outdir, f"{args.label}_{i:04d}.wav")
        print(f"[{i - args.start_index + 1}/{args.count}]")
        try:
            record_one(path)
        except KeyboardInterrupt:
            print("\nStopped early. Resume with:")
            print(f"  python record_samples.py --label {args.label} --count {args.count} "
                  f"--start_index {i} --outdir {args.outdir}")
            return
        # Small pause between recordings, replace by pressing Enter for more control
        input("  Press Enter for next recording (or Ctrl+C to stop)...")

    print(f"\nDone. {args.count} samples saved to {outdir}/")


if __name__ == "__main__":
    main()
