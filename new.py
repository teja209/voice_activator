import os
import subprocess
import glob

def prepare_silence(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(input_dir, '*.*'))
    
    for i, filepath in enumerate(input_files):
        print(f"Slicing room tone: {filepath}")
        output_pattern = os.path.join(output_dir, f"silence_{i}_%04d.wav")
        
        # Segment muxer splits long audio into exact 1.0s chunks
        # Forces 16kHz, mono, 16-bit PCM WAV
        cmd = [
            "ffmpeg", "-y", "-i", filepath,
            "-f", "segment", "-segment_time", "1.0",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {filepath}. Ensure it is a valid audio file.")

if __name__ == "__main__":
    # Point to a folder containing long background noise recordings
    prepare_silence("sound", "sample_processed/silence")