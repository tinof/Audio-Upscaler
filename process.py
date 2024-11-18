import subprocess
import os
from pathlib import Path
import argparse
import torch
from audio_upscaler.inference import AudioSR  # Assuming this is the correct import
import soundfile as sf
import numpy as np
from torch2trt import TRTModule
from trt_utils import convert_to_trt, save_trt_model, load_trt_model

def extract_audio(input_file, temp_dir):
    """Extract audio from video file to WAV format"""
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    cmd = [
        "ffmpeg", "-i", input_file,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # Convert to PCM WAV
        "-ar", "48000",  # Sample rate
        "-ac", "2",  # Stereo
        audio_path
    ]
    subprocess.run(cmd, check=True)
    return audio_path

class AudioProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AudioSR(device=device)
        self.model_trt = None
        self.trt_path = "audiosr_trt.pth"
        
        if device == "cuda":
            self.setup_trt()
    
    def setup_trt(self):
        """Setup TensorRT model"""
        if os.path.exists(self.trt_path):
            # Load existing TRT model
            self.model_trt = TRTModule()
            self.model_trt.load_state_dict(torch.load(self.trt_path))
        else:
            # Convert to TRT
            print("Converting model to TensorRT...")
            self.model_trt = convert_to_trt(self.model)
            save_trt_model(self.model_trt, self.trt_path)
    
    def process_audio(self, audio):
        """Process audio using TensorRT model"""
        if self.model_trt is not None:
            # Use TensorRT model
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).to(self.device)
                if len(audio.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                processed = self.model_trt(audio_tensor).cpu().numpy()
        else:
            # Fallback to regular model
            processed = self.model.enhance(audio)
        return processed

def process_audio(input_audio_path, output_audio_path, processor):
    """Modified process_audio function"""
    audio, sr = sf.read(input_audio_path)
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    if len(audio.shape) == 2:
        left = processor.process_audio(audio[:, 0])
        right = processor.process_audio(audio[:, 1])
        processed_audio = np.stack([left, right], axis=1)
    else:
        processed_audio = processor.process_audio(audio)
    
    sf.write(output_audio_path, processed_audio, 48000)

def remux_audio(input_video, processed_audio, output_file):
    """Remux processed audio with video"""
    # Sony HT-A9 works best with high-bitrate TrueHD or DTS-HD MA
    # Fallback to high-bitrate E-AC3 (DD+) if lossless isn't needed
    cmd = [
        "ffmpeg", "-i", input_video,
        "-i", processed_audio,
        "-c:v", "copy",  # Copy video stream
        "-c:a", "eac3",  # E-AC3 codec
        "-b:a", "1024k",  # High bitrate
        "-ar", "48000",  # 48kHz sampling
        "-channel_layout", "5.1",  # 5.1 channel layout
        "-map", "0:v:0",  # Map video from first input
        "-map", "1:a:0",  # Map audio from second input
        output_file
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Upscale video audio using AudioSR with TensorRT")
    parser.add_argument("input_file", help="Input video file (mkv/mp4)")
    parser.add_argument("--temp_dir", default="temp", help="Temporary directory for processing")
    args = parser.parse_args()

    # Create output filename
    input_path = Path(args.input_file)
    output_file = str(input_path.parent / f"{input_path.stem}_upscaleaudio{input_path.suffix}")

    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AudioProcessor(device=str(device))
        
        # Extract audio
        print("Extracting audio...")
        extracted_audio = extract_audio(args.input_file, args.temp_dir)
        
        # Process audio with TensorRT
        print("Upscaling audio with TensorRT...")
        processed_audio = os.path.join(args.temp_dir, "processed_audio.wav")
        process_audio(extracted_audio, processed_audio, processor)
        
        # Remux
        print("Remuxing video...")
        remux_audio(args.input_file, processed_audio, output_file)

        print(f"Processing complete! Output saved to: {output_file}")

    finally:
        # Cleanup temp files
        if os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)

if __name__ == "__main__":
    main()