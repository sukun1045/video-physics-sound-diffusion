import os
import pickle
import glob

split = 'train'
audio_root = '/data/great_hits/hits_mode_data_ten'
split_root = f'/data/great_hits/{split}.pickle'
rgb_path = f'/data/great_hits/rgb'
select_frames_output_dir = '/data/great_hits/hits_mode_data_video'
os.makedirs(select_frames_output_dir, exist_ok=True)
with open(split_root, 'rb') as handle:
    meta = pickle.load(handle)
file_names = list(meta.keys())
ten_list = ['glass', 'wood', 'ceramic', 'metal', 'cloth',
                         'plastic', 'drywall', 'carpet', 'paper', 'rock']
sr = 44100
fps = 29.97002997002997
save_data = {}
for fn in file_names:
    audio_files = glob.glob(f'{audio_root}/{fn}_*.pickle')
    imgs_list = glob.glob(f'{rgb_path}/{fn}_denoised_thumb/*.jpg')
    imgs_list.sort(key=lambda x: int(x.split('.')[0].split('/')[-1].split('_')[1]))
    for audio_file in audio_files:
        tmp = audio_file.split('/')[-1].split('.')[0].split('_')
        sample = tmp[1]
        material = tmp[2]
        action = tmp[3]
        reaction = tmp[4]
        if action == 'hit' and material in ten_list and reaction == 'static':
            onset_frame = round(float(sample)/sr*fps)
            start_frame = onset_frame - 11
            end_frame = onset_frame + 11
            select_img_list = imgs_list[start_frame:end_frame]
            save_data[f'{fn}_{sample}_{material}_{action}_{reaction}']={'start':start_frame, 'end':end_frame, 'imgs':select_img_list}

with open(f'{select_frames_output_dir}/{split}.pickle', 'wb') as handle:
    pickle.dump(save_data, handle)