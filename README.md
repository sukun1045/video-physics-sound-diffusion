# physics-driven diffusion models for impact sound synthesis from videos (CVPR 2023)

[Project Page](https://sukun1045.github.io/video-physics-sound-diffusion/)

[Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Su_Physics-Driven_Diffusion_Models_for_Impact_Sound_Synthesis_From_Videos_CVPR_2023_paper.pdf)

## Code
Code is available now under the folder [video-physics-sound-diffusion](https://github.com/sukun1045/video-physics-sound-diffusion/tree/main/video-physics-sound-diffusion)!

### Pre-processed data and Pre-trained Weights Links
- [meta data](https://drive.google.com/drive/folders/1Zytmma_OsVF_5HbW_S0TODhiz8tMYunr?usp=drive_link)
- [rgb frames](https://drive.google.com/file/d/19Yyzc_m8aSPffZwwk0LsubGkh00L6mb1/view?usp=drive_link)
- [audio](https://drive.google.com/file/d/1HmMAbxedeJ7fCMHnEFLZVlRJLStVT_xf/view?usp=drive_link)
- [pre-trained residual prediction model](video-physics-sound-diffusion/logs/sound_residual/sound_residual)
- [extracted sound physics and residual parameters](https://drive.google.com/file/d/1lAqp97iNWYTTd0gUahKf8BeMAN1Ap9Z6/view?usp=drive_link)
- [extracted visual features](https://drive.google.com/file/d/1s_xQaLsHeNvrjEk9sl4w2DMAaYfMN_jQ/view?usp=drive_link)
- [extracted physics latents](https://drive.google.com/file/d/11OpVCg5pITjlCautJ4ch7lutA4gW31gT/view?usp=drive_link)
- [pre-trained diffusion model](https://drive.google.com/drive/folders/19lnjIGzVvR5uYZnMjDF0RPNKRrbUTBfg?usp=drive_link)

For questions or help, please open an issue or contact suk4 at uw dot edu

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
- Once done with training, change the `resume_path` in `sound_residual.yaml` to be your model path or use the **pre-trained model** [here](video-physics-sound-diffusion/logs/sound_residual/sound_residual) and you can run `CUDA_VISIBLE_DEVICES=0 python tools/sound_residual_infer.py --cfg configs/sound_residual.yaml` to save both physics and predicted residual parameters as *pickle* file.
- [ ] TODO: Add a jupyter notebook to demonstrate how to reconstruct the sound.

### Training for Physics-driven video to Impact Sound Diffusion
- You must obtain the audio physics and residual parameters before training the diffusion model. 
- We use the visual features extracted from pre-trained resnet 50 + TSM classifier. We provide two types of features: 1) features before the classifier layer are available in [here](https://drive.google.com/drive/folders/1nsT79lghHkQqr9KvEyAHUbQDwsur5kbi?usp=sharing) and the simple lower dimension logits [here](https://drive.google.com/file/d/1s_xQaLsHeNvrjEk9sl4w2DMAaYfMN_jQ/view?usp=drive_link).
- Check the [great_hits_spec_diff.yaml](https://github.com/sukun1045/video-physics-sound-diffusion/blob/main/video-physics-sound-diffusion/configs/great_hits_spec_diff.yaml) and change the data root or other settings if needed.
- Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/great_hits_spec_diff.yaml`

### Generating Samples
- Step 0: change the `resume_path` in `great_hits_spec_diff.yaml` to be your model path.
- Step 1: Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/extract_latents.py --cfg configs/great_hits_spec_diff.yaml` to extract physics latents and save as pickle files.
- Step 2: Under the `video-physics-sound-diffusion` directory, run `CUDA_VISIBLE_DEVICES=0 python tools/query_latents.py --cfg configs/great_hits_spec_diff.yaml` that will use test visual feature to query closest physics latent in training set.
- Step 3: Run `CUDA_VISIBLE_DEVICES=0 python tools/generate_samples.py --configs confings/great_hits_spec_diff.yaml` to generate wave file.
- Using **Pre-trained Model**: Please first download the [processed data](https://drive.google.com/drive/folders/1ara2GL2mA9tcN4e48JJY21xh0DZGNStj?usp=drive_link), then place them under the *data_root* you use in the *config* file. Also, download the [model weights](https://drive.google.com/drive/folders/1rbqOOPJcfsArt69X6vb7A3rWjlqYXcuZ?usp=drive_link) and place it under the *logs* folder. Then, run **Step 3** to generate samples.
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
