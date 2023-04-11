import os

import numpy as np
import librosa
import scipy
import soundfile as sf
import glob
import pickle

def extract_params(x):
    gt_spec = librosa.stft(x, n_fft=2048, hop_length=256)
    abs_spec = np.abs(gt_spec)
    db_gt_spec = librosa.amplitude_to_db(abs_spec, ref=np.max)
    freq_bins = librosa.fft_frequencies(sr=44100, n_fft=2048)  # Frequency components
    psd = np.mean(abs_spec ** 2, axis=1)
    psd_db = 10 * np.log10(psd)
    X = np.fft.rfft(x)
    fft_freqs = np.linspace(0, 44100 / 2, num=len(X))
    Mag_X = 20 * np.log10(np.abs(X))
    final_f = []
    final_a = []
    final_d = []
    for peak, freq in enumerate(psd_db):
        freq_peak = freq_bins[peak]
        power = db_gt_spec[:, 0][peak]
        pick_freq = freq_peak
        large = 0
        for i, fft_freq in enumerate(fft_freqs):
            if abs(fft_freq - freq_peak) <= 10.76 and Mag_X[i] > large:
                large = Mag_X[i]
                pick_freq = fft_freq  # freq_peak
        row = db_gt_spec[peak]
        idx = np.nonzero(row == -80)
        if idx[0].shape[0] == 0:
            frame = 44
        else:
            frame = idx[0][0]
        if frame == 0:
            frame = 0.5
        dcy = frame * 256 / 44100
        final_f.append(pick_freq)
        final_a.append(power)
        final_d.append(dcy)
    final_f = np.vstack(final_f)  # F, 1
    final_a = np.vstack(final_a)
    final_d = np.vstack(final_d)
    angle = 2 * np.pi * final_f  # F,1
    tt = np.arange(0, 11025) / 44100
    mode = np.cos(angle * tt)  # F,T
    amp = (10 ** ((final_a) / 20))
    mode_ = mode * amp
    dcy = np.expand_dims(tt, axis=0) * (60 / final_d)  # F, T x B, F, 1 / 1e3
    env = 10 ** (-dcy / 20)
    mode__ = mode_ * env
    sound = np.sum(mode__, axis=0)  # F,T -> T
    sound = sound / np.max(np.abs(sound))
    return final_f, final_a, final_d, sound

if __name__ == "__main__":
    ten_list = ['glass', 'wood', 'ceramic', 'metal', 'cloth',
                'plastic', 'drywall', 'carpet', 'paper', 'rock']
    audio_dir = '/data/great_hits/audio_data'
    audio_files = glob.glob(f'{audio_dir}/*.pickle')
    save_dir = '/data/great_hits/hits_mode_data_ten'
    os.makedirs(save_dir, exist_ok=True)
    for audio_file in audio_files:
        tmp = audio_file.split('/')[-1].split('.')[0].split('_')
        sample = tmp[1]
        material = tmp[2]
        action = tmp[3]
        reaction = tmp[4]
        if action == 'hit' and material in ten_list and reaction == 'static':
            with open(audio_file, 'rb') as handle:
                gt = pickle.load(handle)
            # print(gt.max(), gt.min())
            gt = gt/np.max(np.abs(gt))
            f, p, t, pred = extract_params(gt)
            fn = f'{tmp[0]}_{sample}_{material}_{action}_{reaction}'
            print(fn)
            data_dict = {'fn': fn, 'f': f, 'p': p, 't': t,
                         'gt': gt, 'pred': pred}
            with open(f'{save_dir}/{fn}.pickle', 'wb') as handle:
                pickle.dump(data_dict, handle)
