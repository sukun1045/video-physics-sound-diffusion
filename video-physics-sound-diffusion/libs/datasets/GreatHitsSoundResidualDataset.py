import glob
from torch.utils.data import Dataset
import numpy as np
import pickle
import librosa
import librosa.display
import math

class GreatHitsDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(GreatHitsDataset, self).__init__()
        self.audio_root = data_root + 'hits_mode_data_ten'
        self.split = split
        self.split_root = data_root + f'{self.split}.pickle'
        self.ten_list = ['glass', 'wood', 'ceramic', 'metal', 'cloth',
                         'plastic', 'drywall', 'carpet', 'paper', 'rock']
        self.data = []
        self.load_data()

    def load_data(self):
        with open(self.split_root, 'rb') as handle:
            meta = pickle.load(handle)
        file_names = list(meta.keys())
        for fn in file_names:
            audio_files = glob.glob(f'{self.audio_root}/{fn}_*.pickle')
            for audio_file in audio_files:
                tmp = audio_file.split('/')[-1].split('.')[0].split('_')
                sample = tmp[1]
                material = tmp[2]
                action = tmp[3]
                reaction = tmp[4]
                if action == 'hit' and material in self.ten_list and reaction == 'static':
                    self.data.append(audio_file)
        print('{} data number: {}'.format(self.split, len(self.data)))

    def __getitem__(self, index):
        data_file = self.data[index]
        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
        gt = data['gt']
        pred = data['pred']
        gt_spec = np.abs(librosa.stft(gt, n_fft=2048, hop_length=256))
        return gt, pred, gt_spec

    def __len__(self):
        return len(self.data)

