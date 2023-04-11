import glob
from torch.utils.data import Dataset
import numpy as np
import pickle
import librosa
import librosa.display

class GreatHitsDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(GreatHitsDataset, self).__init__()
        self.video_root = data_root + f'hits_100_mode_data_ten_video/{split}.pickle'
        self.video_fea_root = data_root + f'hits_ten_mat_video_classifer_all_features/{split}.pickle'
        self.audio_root = data_root + 'hits_100_mode_data_ten'
        self.latent_root = data_root + f'hits_mode_data_ten_physics_noise_params/{split}.pickle'
        self.split = split
        self.data = []
        self.load_data()
        self.spec_max = 5.9540715
        self.spec_min = -18.420681
    def load_data(self):
        with open(self.video_root, 'rb') as handle:
            video_data = pickle.load(handle)
        with open(self.video_fea_root, 'rb') as handle:
            video_features = pickle.load(handle)
        freq_bins = librosa.fft_frequencies(sr=44100, n_fft=2048)
        freq_resolution = freq_bins[1] - freq_bins[0]
        t_min = 0.5 * 256 / 44100
        t_max = 44 * 256 / 44100
        with open(self.latent_root, 'rb') as handle:
            raw_data = pickle.load(handle)
        for fn in list(video_data.keys()):
            v = video_data[fn]
            f = raw_data[fn]['f'].squeeze()
            f_delta = f - freq_bins
            f_norm = 2 * f_delta / freq_resolution
            p_norm = 2 * raw_data[fn]['p'].squeeze() / (-80) - 1
            t_norm = (raw_data[fn]['t'].squeeze() - t_min) / (t_max - t_min)
            t_norm = t_norm * 2 - 1
            noise_weights = raw_data[fn]['noise_weights']
            noise_t_norm = (raw_data[fn]['noise_t'] - 1e-5) / 0.5
            video_fea = video_features[fn]
            data = {'fn': fn, 'f': f_norm, 'p': p_norm, 't': t_norm,
                    'noise_weights': noise_weights, 'noise_t': noise_t_norm,
                    'start': v['start'], 'end': v['end'], 'imgs': v['imgs'],
                    'video_feature':np.mean(video_fea, axis=0)}
            self.data.append(data)
        print('{} data number: {}'.format(self.split, len(self.data)))

    def __getitem__(self, index):
        data = self.data[index]
        file = data['fn']
        with open(f'{self.audio_root}/{file}.pickle', 'rb') as handle:
            audio_data = pickle.load(handle)
        audio_file = audio_data['file']
        raw_audio = np.load(audio_file, allow_pickle=True)
        audio = raw_audio / np.max(np.abs(raw_audio))
        spec = np.log(np.abs(librosa.stft(audio, n_fft=2048, hop_length=256)) + 1e-8)
        spec = 2*(spec - self.spec_min)/(self.spec_max - self.spec_min) - 1
        f = data['f']
        p = data['p']
        t = data['t']
        noise_weights = data['noise_weights']
        noise_t = data['noise_t']
        video_fea = data['video_feature']
        return f, p, t, noise_weights, noise_t, video_fea, spec[None, ...]

    def __len__(self):
        return len(self.data)
