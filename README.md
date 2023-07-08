# physics-driven diffusion models for impact sound synthesis from videos (CVPR 2023)

[Project Page](https://sukun1045.github.io/video-physics-sound-diffusion/)

[Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Su_Physics-Driven_Diffusion_Models_for_Impact_Sound_Synthesis_From_Videos_CVPR_2023_paper.pdf)

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
- Use [video_to_frames.py](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video_to_frames.py) to extract rgb frames from video and save as 224x224 images. Processed rgb frames are available in [Google Drive](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing) (zip file name: **rgb**).
- Use [video_to_wav.py](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/tools/video_to_wavs.py) to extract impact sound segments from videos. Extracted audio files are available in [Google Drive](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing) (zip file name: **audio_data**).
- Use [extract_physics_params.py](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/tools/extract_physics_params.py) to extract physics parameters from audio and save freq, power, decay rate, ground truth audio, and reconstructed audio as pickle file.
- Use [train_test_split_process_video_data.py](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/tools/train_test_split_process_video_data.py) to segment the video frames and save train/test meta files. Processed meta files are in [Google Drive](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing) (**segmented_video_data_split**).
- Note: we mainly use the subset annotated with **hits** action and **static** reaction. This ends up at ten representative classes of materials (glass, wood, ceramic, metal, cloth, plastic, drywall, carpet, paper, and rock) in a total of 10k impact sounds. You could also try using all sounds available in the dataset. While the annotations are noisy, we find that using the physics + residual combination can still reconstruct the audio reasonably.

### Training and Inference for Sound Physics and Residual Prediction
- Check the [sound_residual.yaml](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/configs/sound_residual.yaml) and change the data root or other settings if needed.
- Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/sound_residual_train.py --configs configs/sound_residual.yaml`
- Once done with training, change the `resume_path` in `sound_residual.yaml` to be your model path and you can run `CUDA_VISIBLE_DEVICES=0 python tools/sound_residual_infer.py --configs confings/sound_residual.yaml` to save both physics and predicted residual parameters as pickle file.
- Predicted physics and residual parameters are available in [Google Drive](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing).
- [ ] TODO: Add a jupyter notebook to demonstrate how to reconstruct the sound.

### Training for Physics-driven video to Impact Sound Diffusion
- We use visual features extracted from pre-trained resnet 50 + TSM. Processed visual features are available in [Google Drive](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing).
- Check the [great_hits_spec_diff.yaml](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/configs/great_hits_spec_diff.yaml) and change the data root or otther settings if needed.
- Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/train.py --configs configs/great_hits_spec_diff.yaml`

### Generating Samples
- change the `resume_path` in `great_hits_spec_diff.yaml` to be your model path.
- Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/extract_latents.py --cfg configs/great_hits_spec_diff.yaml` to extract physics latents and save as pickle files.
- Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/query_latents.py --cfg configs/great_hits_spec_diff.yaml` that will use test visual feature to query closest physics latent in training set.
- Run `CUDA_VISIBLE_DEVICES=0 python tools/generate_samples.py --configs confings/great_hits_spec_diff.yaml` to generate wave file.
- [ ] TODO: Add a jupyter notebook for an easier demo.

# Citation
If you find this repo useful for your research, please consider citing the paper
> @inproceedings{su2023physics,
  title={Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos},
  author={Su, Kun and Qian, Kaizhi and Shlizerman, Eli and Torralba, Antonio and Gan, Chuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9749--9759},
  year={2023}
}
# Acknowledgements
Part of the code is borrowed from the following repo and we would like to thank the authors for their contribution.
- https://github.com/lucidrains/denoising-diffusion-pytorch
  
We would like to thank the authors of <cite><a href="https://andrewowens.com/vis/">the Greatest Hits dataset</a></cite> for making this dataset possible.
		We would like to thank <cite><a href="https://vinayak-agarwal.com/">Vinayak Agarwal</a></cite> for his suggestions on physics mode parameters estimation from raw audio.
		We would like to thank the authors of <cite><a href="https://sites.google.com/view/diffimpact">DiffImpact</a></cite> for inspiring us to use the physics-based sound synthesis method to design physics priors as a conditional signal to guide the deep generative model synthesizes impact sounds from videos.
