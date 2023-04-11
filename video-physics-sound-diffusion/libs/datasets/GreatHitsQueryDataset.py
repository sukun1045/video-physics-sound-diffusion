import glob
from torch.utils.data import Dataset
import numpy as np
import pickle
import librosa
import librosa.display
import math
from PIL import Image
import torchvision.transforms as transforms
class GreatHitsDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(GreatHitsDataset, self).__init__()
        self.video_root = data_root + f'hits_100_mode_data_ten_video/{split}.pickle'
        self.video_fea_root = data_root + f'hits_ten_mat_video_classifer_all_features/{split}.pickle'
        self.audio_root = data_root + 'hits_100_mode_data_ten'
        self.latent_root = data_root + f'hits_ten_mat_spec_diff_params_latent_video_fea_latents/query_{split}.pickle'
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
        with open(self.latent_root, 'rb') as handle:
            raw_data = pickle.load(handle)
        for fn in list(video_data.keys()):
            v = video_data[fn]
            latent= raw_data[fn]
            video_fea = video_features[fn]
            data = {'fn': fn, 'latent':latent,
                    'start': v['start'], 'end': v['end'], 'imgs': v['imgs'],
                   'video_feature':video_fea}
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
        latent = data['latent']
        video_fea = data['video_feature']
        return latent, video_fea, spec[None, ...]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_root = '/data/great_hits/'
    dataset = GreatHitsDataset(data_root, split='test')
    _, video_fea, _ = dataset.__getitem__(0)
    print(video_fea.shape)

