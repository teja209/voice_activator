"""
Convert teammates' phone recordings into the 16kHz mono 16-bit WAV format needed
for training. Handles two recording styles (see RECORDING STYLES below).

Requires ffmpeg installed:
    Mac:   brew install ffmpeg
    Linux: sudo apt install ffmpeg
    Win:   choco install ffmpeg   (or download from ffmpeg.org and add to PATH)

Usage (Style A -- one file per utterance, e.g. 50 separate voice memos):
    python convert_phone_recordings.py --mode separate \
        --input_dir raw_phone_recordings/priya --label marvin --outdir data/priya

Usage (Style B -- one long recording, keyword repeated with pauses, auto-split):
    python convert_phone_recordings.py --mode split_long \
        --input_file raw_phone_recordings/priya_long.m4a --label marvin --outdir data/priya
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found. Install it first:")
        print("  Mac:   brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        print("  Windows: choco install ffmpeg  (or download from ffmpeg.org)")
        sys.exit(1)


def convert_single(input_path, output_path, target_duration=1.0):
    """Convert any audio file to 16kHz mono 16-bit WAV, padded/trimmed to target_duration."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000",           # sample rate
        "-ac", "1",               # mono
        "-sample_fmt", "s16",     # 16-bit
        "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
        str(output_path)
    ], capture_output=True, check=True)


def mode_separate(input_dir, label, outdir):
    """Each file in input_dir is one utterance of the keyword."""
    input_dir = Path(input_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exts = (".m4a", ".aac", ".mp3", ".wav", ".caf", ".ogg", ".opus")
    files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in exts])

    if not files:
        print(f"No audio files found in {input_dir} (looked for {exts})")
        return

    print(f"Converting {len(files)} files...")
    for i, f in enumerate(files):
        out_path = outdir / f"{label}_{i:04d}.wav"
        try:
            convert_single(f, out_path)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED on {f.name}: {e.stderr.decode()[:200]}")
            continue
    print(f"Done. {len(files)} files converted into {outdir}/")


def mode_split_long(input_file, label, outdir, silence_thresh="-35dB", min_silence="0.3"):
    """
    One long recording with the keyword repeated, separated by pauses.
    Uses ffmpeg's silencedetect to find gaps and split into individual 1s clips.
    Good for teammates who'd rather just say the word 50x in one go with pauses
    than manage 50 separate voice memos.
    """
    input_file = Path(input_file)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # First convert the whole thing to 16kHz mono WAV
    tmp_wav = outdir / "_full_recording_tmp.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_file),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(tmp_wav)
    ], capture_output=True, check=True)

    # Detect silence gaps to find split points
    result = subprocess.run([
        "ffmpeg", "-i", str(tmp_wav),
        "-af", f"silencedetect=noise={silence_thresh}:d={min_silence}",
        "-f", "null", "-"
    ], capture_output=True, text=True)

    # Parse silence_start/silence_end timestamps from stderr (ffmpeg logs filter output there)
    import re
    silence_starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    silence_ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", result.stderr)]

    import wave
    with wave.open(str(tmp_wav)) as wf:
        total_dur = wf.getnframes() / wf.getframerate()

    if not silence_ends:
        print("No silence gaps detected -- try adjusting --silence_thresh or check the recording")
        print("has actual pauses between repetitions. Falling back to fixed 1.2s windows.")
        # crude fallback: just chop into fixed windows
        starts = [i * 1.2 for i in range(int(total_dur / 1.2))]
    else:
        # A word lives in the gap AFTER each silence_end and BEFORE the next silence_start.
        # Start each clip just after silence ends (small offset forward, not backward --
        # going backward lands the clip still inside the silent gap, which was the bug).
        starts = [min(e + 0.05, total_dur - 0.1) for e in silence_ends]
        # If the recording doesn't end in silence, the last word (after the final
        # silence_end, before end-of-file) is still captured by this -- no dropped word.

    print(f"Found {len(starts)} candidate utterances, extracting 1s clips...")
    count = 0
    for start in starts:
        out_path = outdir / f"{label}_{count:04d}.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp_wav),
                "-ss", str(start), "-t", "1.0",
                "-af", "apad=whole_dur=1.0",
                str(out_path)
            ], capture_output=True, check=True)
            # Verify the output actually has audio -- catches any silent/empty extraction
            with wave.open(str(out_path)) as check_wf:
                if check_wf.getnframes() < 16000:
                    print(f"  WARNING: {out_path.name} shorter than expected "
                          f"({check_wf.getnframes()} frames) -- may be near end of recording")
            count += 1
        except (subprocess.CalledProcessError, wave.Error) as e:
            print(f"  FAILED extracting clip at {start:.2f}s: {e}")
            continue

    tmp_wav.unlink()

    # Post-run check: flag any clips that are empty/near-silent so they don't
    # silently make it into training (this is exactly what went wrong before this fix)
    import array
    empty_or_silent = []
    for i in range(count):
        p = outdir / f"{label}_{i:04d}.wav"
        try:
            with wave.open(str(p)) as wf:
                frames = wf.readframes(wf.getnframes())
                if len(frames) == 0:
                    empty_or_silent.append((p.name, "empty"))
                    continue
                samples = array.array('h', frames)
                max_amp = max(abs(s) for s in samples) if samples else 0
                if max_amp < 200:
                    empty_or_silent.append((p.name, f"near-silent (max_amp={max_amp})"))
        except Exception as e:
            empty_or_silent.append((p.name, f"unreadable: {e}"))

    print(f"Extracted {count} clips into {outdir}/")
    if empty_or_silent:
        print(f"\nWARNING: {len(empty_or_silent)}/{count} clips look empty or silent -- "
              f"these likely missed the word and should NOT go into training as-is:")
        for name, reason in empty_or_silent:
            print(f"  {name}: {reason}")
        print("Try a lower --silence_thresh (e.g. -40dB) if most clips are flagged,")
        print("or just delete the flagged files if it's only a few.")
    else:
        print("All clips have audio content (no empty/silent extractions detected).")
    print("\nSTILL listen to a handful manually -- this check catches total silence,")
    print("not whether the split landed cleanly on the word vs. cutting it off.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["separate", "split_long"], required=True)
    ap.add_argument("--input_dir", help="for --mode separate: folder of individual recordings")
    ap.add_argument("--input_file", help="for --mode split_long: one long recording file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--silence_thresh", default="-35dB", help="for split_long: silence detection threshold")
    ap.add_argument("--min_silence", default="0.3", help="for split_long: min silence gap duration (sec)")
    args = ap.parse_args()

    check_ffmpeg()

    if args.mode == "separate":
        if not args.input_dir:
            print("ERROR: --input_dir required for --mode separate")
            sys.exit(1)
        mode_separate(args.input_dir, args.label, args.outdir)
    else:
        if not args.input_file:
            print("ERROR: --input_file required for --mode split_long")
            sys.exit(1)
        mode_split_long(args.input_file, args.label, args.outdir, args.silence_thresh, args.min_silence)


if __name__ == "__main__":
    main()