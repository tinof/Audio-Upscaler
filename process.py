import subprocess
import os
from pathlib import Path
import argparse
import torch
import soundfile as sf
import numpy as np
from torch2trt import TRTModule
from trt_utils import convert_to_trt, save_trt_model, load_trt_model
from audio_upscaler import upscale
from audio_upscaler.predict import Predictor

def extract_audio(input_file, temp_dir):
    """Extract audio from video file to WAV format"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    cmd = [
        "ffmpeg", "-i", input_file,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # Convert to PCM WAV
        "-ar", "48000",  # Sample rate
        "-ac", "2",  # Stereo
        audio_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

class AudioProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.predictor = Predictor()
        self.predictor.setup(model_name="basic", device=device)
        self.model = self.predictor
        self.model_trt = None
        self.trt_path = "audiosr_trt.pth"
        
        if device == "cuda":
            self.setup_trt()
    
    def setup_trt(self):
        """Setup TensorRT model with proper error handling"""
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return

        try:
            if os.path.exists(self.trt_path):
                print("Loading existing TensorRT model...")
                self.model_trt = load_trt_model(self.trt_path)
            else:
                print("Converting model to TensorRT...")
                self.model_trt = convert_to_trt(self.model)
                save_trt_model(self.model_trt, self.trt_path)
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
            self.model_trt = None
    
    def _process_chunk(self, chunk):
        """Process a single chunk of audio"""
        # Convert to tensor and move to device
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
        
        # Process with TensorRT if available, otherwise use original model
        with torch.no_grad():
            if self.model_trt is not None:
                try:
                    processed = self.model_trt(chunk_tensor)
                except Exception as e:
                    print(f"TensorRT inference failed, falling back to original model: {e}")
                    processed = self.model(chunk_tensor)
            else:
                processed = self.model(chunk_tensor)
        
        return processed.cpu().numpy().squeeze(0)
    
    def process_audio(self, audio, chunk_size=480000):
        """Process audio in chunks to avoid memory issues"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Process in chunks if audio is too long
        if audio.shape[-1] > chunk_size:
            chunks = []
            for i in range(0, audio.shape[-1], chunk_size):
                chunk = audio[..., i:i + chunk_size]
                processed_chunk = self._process_chunk(chunk)
                chunks.append(processed_chunk)
            return np.concatenate(chunks, axis=-1)
        
        return self._process_chunk(audio)

def process_audio(input_audio_path, output_audio_path, processor):
    """Process audio file"""
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")
    
    audio, sr = sf.read(input_audio_path)
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    if len(audio.shape) == 2:
        left = processor.process_audio(audio[:, 0])
        right = processor.process_audio(audio[:, 1])
        processed_audio = np.stack([left, right], axis=1)
    else:
        processed_audio = processor.process_audio(audio)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    sf.write(output_audio_path, processed_audio, 48000)

def remux_audio(input_video, processed_audio, output_file):
    """Remux processed audio with video"""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not os.path.exists(processed_audio):
        raise FileNotFoundError(f"Processed audio not found: {processed_audio}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cmd = [
        "ffmpeg", "-i", input_video,
        "-i", processed_audio,
        "-c:v", "copy",  # Copy video stream
        "-c:a", "eac3",  # E-AC3 codec
        "-b:a", "1024k",  # High bitrate
        "-ar", "48000",  # 48kHz sampling
        "-channel_layout", "stereo",  # Changed to stereo for compatibility
        "-map", "0:v:0",  # Map video from first input
        "-map", "1:a:0",  # Map audio from second input
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg remux failed: {e.stderr.decode()}")

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

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        # Cleanup temp files
        if os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)

if __name__ == "__main__":
    main()