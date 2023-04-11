import os
import glob
import pandas as pd
import numpy as np
import librosa
import pickle

wav_files = glob.glob('/Users/kunsu/Documents/great_hits/vis-data-256/*_denoised.wav')
print(len(wav_files))
sr = 44100
dur = 0.25
count = 0
save_dir = '/Users/kunsu/Documents/great_hits/audio_data'
os.makedirs(save_dir, exist_ok=True)
for wav_file in wav_files:
    wav, _ = librosa.load(wav_file, sr=sr)
    name = wav_file.split('/')[-1].split('.')[0].split('_')[0]
    # print(name)
    label_file = f'/Users/kunsu/Documents/great_hits/vis-data-256/{name}_times.txt'
    data = pd.read_csv(label_file, sep=" ", header=None)
    data.columns = ["time", "material", "action", "reaction"]
    time_stamps = data.time.shape[0]
    # print('num of hits', time_stamps)
    onset_samples = []
    for i in range(time_stamps):
        t = data.time[i]
        if i == 0:
            prev_t = t
        else:
            if t - prev_t < 0.3:
                print('MORE THAN 1 SOUND WITHIN 0.25 Sec')
                count += 1
                prev_t = t
        sample = round(t * sr)
        material = data.material[i]
        action = data.action[i]
        reaction = data.reaction[i]
        if sample + int(sr*dur) > wav.shape[0]:
            wav_seg = np.zeros(int(sr*dur))
            tmp = wav[sample:]
            wav_seg[:tmp.shape[0]] = tmp
        else:
            wav_seg = wav[sample:sample+int(sr*dur)]
        pickle_file = f'{save_dir}/{name}_{sample}_{material}_{action}_{reaction}.pickle'
        with open(pickle_file, 'wb') as handle:
            pickle.dump(wav_seg, handle)
print(count)

