"""
Merge multiple teammates' recorded-keyword zips into a single clean dataset folder,
ready to zip and upload to the training notebook.

Usage:
    python merge_recordings.py --zips priya_keyword.zip rahul_keyword.zip me_keyword.zip \
                                --outdir merged_data --label marvin

What it does:
    - Extracts each zip
    - Finds all .wav files inside (regardless of nested folder structure)
    - Renames them sequentially into one folder: marvin_0000.wav, marvin_0001.wav, ...
    - Reports total count and flags any files that aren't 16kHz mono (common gotcha
      if someone recorded with a different tool than record_samples.py)
    - Prints a couple of basic sanity stats (duration, silence check) so you can catch
      obviously broken recordings before they go into training

Then zip the output folder and upload that single zip to the notebook (Cell 3).
"""
import argparse
import os
import shutil
import zipfile
import tempfile
import wave
import contextlib
from pathlib import Path


def find_wavs(root):
    return list(Path(root).rglob("*.wav"))


def check_wav(path):
    """Returns (sample_rate, channels, duration_sec, is_silent) or None if unreadable."""
    try:
        with contextlib.closing(wave.open(str(path), 'rb')) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n_frames = wf.getnframes()
            duration = n_frames / float(sr)
            frames = wf.readframes(n_frames)
            # crude silence check: is everything near-zero?
            import array
            samples = array.array('h', frames) if wf.getsampwidth() == 2 else None
            is_silent = False
            if samples:
                max_amp = max(abs(s) for s in samples[:min(len(samples), sr)])
                is_silent = max_amp < 200  # near-zero threshold
            return sr, ch, duration, is_silent
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True, help="paths to teammates' zip files")
    ap.add_argument("--outdir", default="merged_data")
    ap.add_argument("--label", required=True, help="keyword label, used as filename prefix")
    args = ap.parse_args()

    out_root = Path(args.outdir) / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    counter = 0
    problems = []
    stats = {"total": 0, "wrong_rate": 0, "wrong_channels": 0, "possibly_silent": 0, "short_or_long": 0}

    with tempfile.TemporaryDirectory() as tmp:
        for zip_path in args.zips:
            zip_path = Path(zip_path)
            if not zip_path.exists():
                print(f"WARNING: {zip_path} not found, skipping")
                continue

            extract_dir = Path(tmp) / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_dir)

            wavs = find_wavs(extract_dir)
            print(f"{zip_path.name}: found {len(wavs)} wav files")

            for wav in wavs:
                stats["total"] += 1
                info = check_wav(wav)
                if info is None:
                    problems.append(f"  UNREADABLE: {zip_path.name} -> {wav.name}")
                    continue
                sr, ch, duration, is_silent = info
                flag = []
                if sr != 16000:
                    stats["wrong_rate"] += 1
                    flag.append(f"sample_rate={sr}")
                if ch != 1:
                    stats["wrong_channels"] += 1
                    flag.append(f"channels={ch}")
                if duration < 0.5 or duration > 1.5:
                    stats["short_or_long"] += 1
                    flag.append(f"duration={duration:.2f}s")
                if is_silent:
                    stats["possibly_silent"] += 1
                    flag.append("looks silent/near-empty")

                if flag:
                    problems.append(f"  {zip_path.name} -> {wav.name}: {', '.join(flag)}")

                dest_name = f"{args.label}_{counter:04d}.wav"
                shutil.copy(wav, out_root / dest_name)
                counter += 1

    print(f"\nMerged {counter} files into {out_root}/")
    print(f"\nStats: {stats}")

    if problems:
        print(f"\n{len(problems)} files flagged for review (not removed, just flagged -- "
              f"listen to a few of these before trusting them):")
        for p in problems[:30]:
            print(p)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
    else:
        print("\nNo issues detected. Looks clean.")

    print(f"\nNext step:")
    print(f"  cd {args.outdir}")
    print(f"  zip -r ../{args.label}_merged.zip {args.label}/")
    print(f"  Then upload {args.label}_merged.zip to the training notebook (Cell 3).")


if __name__ == "__main__":
    main()
