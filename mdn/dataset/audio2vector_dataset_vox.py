import torch.utils.data as data
import os, glob, platform
import numpy as np
import cv2
import torch
import sys
sys.path.append('.')
sys.path.append('..')
from torch.utils.data._utils.collate import default_collate
from sklearn.utils import shuffle

from scipy.io import wavfile as wav
from scipy.signal import stft
from scipy import signal
from scipy.signal import get_window
import pandas as pd
from librosa.filters import mel
import random
import glob
from numpy.random import RandomState
import soundfile as sf

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=320, hop_length=160):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)


class audio_vector_dataset_self_register_multi(data.Dataset):

    def __init__(self, model):

        self.src_dir = r' '
        self.status = model

        self.df = pd.read_csv(os.path.join(self.src_dir, model + '.csv'))
        self.df = self.df.sample(frac=1, random_state=20).reset_index(drop=True)
        if self.status == 'train':
            self.inteval = 1
        elif self.status == 'dev':
            self.inteval = 1
        print('Loading Data_{}'.format(self.status))

        print(self.df.shape[0])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        """
        Get landmark alignment outside in train_pass()
        """
        row = self.df.loc[item]

        arr = np.load(row['pose_fn'])

        imgs_path = arr['imgs']
        img_path0 = imgs_path[0]
        
        arr_angle_dis = np.load(row['pose_fn'].replace('npz/', 'npz_smooth/'), allow_pickle=True)

        name =  img_path0.split('/')[-4] + '_' + img_path0.split('/')[-3] + '_' + img_path0.split('/')[-2] + '_' + img_path0.split('/')[-1][:-4].replace('.', '_')
        

        pid = np.array([0] * 60)

        # TODO
        fl_data = arr_angle_dis['anchor_t_shape']
        trans_data = arr_angle_dis['rot_trans']
        angle_data = arr_angle_dis['rot_quats']
        length = imgs_path.shape[0]

        # audio
        audio_path = str(arr_angle_dis['audio'])
        
        samples, sample_rate = sf.read(audio_path)
        

        D = pySTFT(samples).T
        mel_basis = mel(16000, 320, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)  

        start = row['start']
        end = row['end']
        audio_in = []
        lmks = []
        rot_trans = []
        rot_quats = []
        for item in range(start + 2, end - 2, self.inteval):
            sel_audio_clip = S[(item - 2) * 4:(item + 3) * 4, :]
            lmks.append(fl_data[item - start])
            rot_quats.append(angle_data[item - start])
            rot_trans.append(trans_data[item - start])
            for _ in range(20 - sel_audio_clip.shape[0]):
                sel_audio_clip = np.concatenate((sel_audio_clip, sel_audio_clip[-1:]), axis=0)
                print('*****************************')
            assert sel_audio_clip.shape[0] == 20
            audio_in.append(sel_audio_clip)
        lmks = np.stack(lmks, axis=0)
        lmks = lmks.reshape(lmks.shape[0], 136)
        rot_trans = np.stack(rot_trans, axis=0)
        rot_quats = np.stack(rot_quats, axis=0)
        audio_in = np.stack(audio_in, axis=0).astype(np.float32)

        assert audio_in.shape[0] == lmks.shape[0]

        return audio_in, lmks, samples[int(start * sample_rate / 25): int(end * sample_rate / 25)], name, rot_trans, rot_quats, pid

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)
    
    def preprocess_x(self, x):
        if len(x) > 67267:
            x = x[:67267]
        elif len(x) < 67267:
            x = np.pad(x, [0, 67267 - len(x)], mode='constant', constant_values=0)
        return x
    
