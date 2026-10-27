import os
import subprocess
import glob

def prepare_unknown(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(input_dir, '*.*'))
    
    for i, filepath in enumerate(input_files):
        output_file = os.path.join(output_dir, f"unknown_{i:04d}.wav")
        
        filters = (
            "silenceremove=start_periods=1:start_duration=0:start_threshold=-35dB,"
            "apad=whole_dur=1.0,"
            "atrim=end=1.0"
        )
        
        cmd = [
            "ffmpeg", "-y", "-i", filepath,
            "-af", filters,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            output_file
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"Processed 1.0s unknown sample: {output_file}")
        except subprocess.CalledProcessError:
            print(f"Error processing {filepath}")

if __name__ == "__main__":
    # Point to a folder of individual short speech clips
    prepare_unknown("dataset_raw/real_unknown", "dataset/unknown")