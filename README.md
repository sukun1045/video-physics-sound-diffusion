# physics-driven diffusion models for impact sound synthesis from videos (CVPR 2023)

[Project Page](https://sukun1045.github.io/video-physics-sound-diffusion/)

[Paper Link](https://arxiv.org/abs/2303.16897)

## Code
Code is available now under the folder [video-physics-sound-diffusion](https://github.com/sukun1045/video-physics-sound-diffusion/tree/main/video-physics-sound-diffusion)!

### Requirement
- python=3.9
- pytorch=1.12.0
- torchaudio=0.12.0
- librosa=0.9.2
- auraloss=0.2.2
- einops=0.4.1
- einops-exts=0.0.3
- numpy=1.22.3
- opencv-python=4.6.0.66

### Prepare Data
- Download the [Greatest Hits dataset](https://andrewowens.com/vis/) videos and metadata (txt files).
- The dataset has the following stats:
  - tot sound:  46577
  - material stats: {'gravel': 437, 'dirt': 3279, 'rock': 2795, 'None': 18133, 'wood': 4587, 'cloth': 2085, 'metal': 4118, 'paper': 1802, 'grass': 981, 'leaf': 2515, 'plastic': 2176, 'tile': 349, 'drywall': 698, 'plastic-bag': 440, 'carpet': 377, 'glass': 382, 'water': 986, 'ceramic': 437}
  - action stats: {'hit': 19619, 'None': 17942, 'scratch': 9016}
- Use [video to frames script](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video_to_frames.py) to extract rgb frames from video and save as 224x224 images.
- Use [video to wav script](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/tools/video_to_wavs.py) to extract impact sound segments from videos.
- Extract Physics Parameters from audio using [this script](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/tools/extract_physics_params.py).
- Preprocessed data will be available soon!!

### Training
- 
More instructions are coming.
