# Audio Upscaler (AudioSR) with TensorRT Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2309.07314-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2309.07314)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audiosr)

## Overview

AudioSR is a powerful tool designed to enhance the fidelity of your audio files, optimized for NVIDIA RTX GPUs using TensorRT. This fork adds TensorRT support and video processing capabilities, making it especially suitable for upscaling audio tracks in video files.

## Key Features

- **TensorRT Optimization**: Leverages NVIDIA TensorRT for faster processing on RTX GPUs
- **Video Support**: Direct processing of MKV/MP4 files with automatic audio extraction and remuxing
- **High Fidelity**: Produces high-quality output with enhanced clarity and detail
- **Versatility**: Works with all types of audio content (music, speech, environmental sounds)
- **Home Theater Optimization**: Configured for optimal output with modern AV receivers

## Installation (Windows 11)

1. **Install Miniforge3**:
   - Download from [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
   - Choose `Miniforge3-Windows-x86_64.exe`
   - Run installer (select "Add to PATH")

2. **Setup Environment**:

```bash
conda create -n audiosr python=3.11
conda activate audiosr
conda install conda-forge::cudatoolkit
conda install cudnn ffmpeg -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install nvidia-pyindex nvidia-tensorrt

# Install torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

## Usage

1. **Activate Environment**:
```bash
conda activate audiosr
```

2. **Process Video File**:
```bash
python process.py input_video.mkv output_video.mkv
```

## Requirements

- Windows 11/10
- NVIDIA GPU (memory management optimized for RTX series)
- CUDA Toolkit 11.8
- TensorRT
- FFmpeg

## Performance

- Optimized for RTX GPUs using TensorRT
- 2-4x speedup compared to base implementation
- FP16 precision support
- Efficient memory usage

## Common Issues

If CUDA is not found, add to system environment variables:
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
PATH += %CUDA_PATH%\bin
```

## Changes from Original

- Added TensorRT support
- Added video processing capabilities
- Optimized for RTX GPUs
- Added batch processing support
- Improved memory management

## Acknowledgments
Based on the work of https://github.com/haoheliu/versatile_audio_super_resolution/

```bibtex
@article{liu2023audiosr,
  title={{AudioSR}: Versatile Audio Super-resolution at Scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2309.07314},
  year={2023}
}
```

## License

This project maintains the same license as the original AudioSR repository.

## 🚀 Roadmap & To-Do

### High Priority
- [ ] Implement TensorRT engine caching for faster startup
- [ ] Add batch processing for multiple files
- [ ] Implement progress bar for long processing tasks
- [ ] Add support for different output audio codecs (TrueHD, DTS-HD MA)

### Audio Processing
- [ ] Add pre-processing noise reduction option
- [ ] Implement adaptive chunk size based on available VRAM
- [ ] Add support for multichannel audio (5.1, 7.1)
- [ ] Implement seamless chunk boundary processing

### Performance Optimization
- [ ] Optimize TensorRT conversion parameters for RTX 40 series
- [ ] Add dynamic batch sizing based on GPU capabilities
- [ ] Implement memory usage monitoring
- [ ] Add support for multiple GPU processing

### Features
- [ ] Add GUI interface
- [ ] Implement A/B comparison tool
- [ ] Add audio preview functionality
- [ ] Create presets for different use cases (movies, music, speech)

### Quality of Life
- [ ] Add configuration file support
- [ ] Implement logging system
- [ ] Add error recovery for long processing tasks
- [ ] Create detailed documentation for all features

### Testing & Validation
- [ ] Add automated tests
- [ ] Create benchmark suite
- [ ] Implement quality metrics reporting
- [ ] Add validation for different GPU models

### Documentation
- [ ] Create detailed API documentation
- [ ] Add examples for common use cases
- [ ] Create troubleshooting guide
- [ ] Add performance optimization guide

### Future Ideas
- [ ] Investigate INT8 quantization support
- [ ] Research adaptive quality settings
- [ ] Consider implementing CUDA graphs
- [ ] Explore DirectML support for AMD GPUs

Feel free to contribute to any of these items! Pull requests are welcome.
