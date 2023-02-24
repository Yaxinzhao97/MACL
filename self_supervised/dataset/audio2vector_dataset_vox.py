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
import librosa
from librosa.filters import mel
import random
import glob
from numpy.random import RandomState

import soundfile as sf

from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

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

class audio_vector_dataset_vox_multi(data.Dataset):

    def __init__(self, model, transform, num_frames=1):

        self.src_dir = r' '
        self.status = model
        
        self.transform = transform
        with open(os.path.join(self.src_dir, self.status + '.txt'), 'r') as f:
            self.filenames = f.readlines()

        valid_idx = list(range(len(self.filenames)))
        
        random.seed(0)
        random.shuffle(valid_idx)
        self.filenames = [self.filenames[i] for i in valid_idx]
        if model == 'train':
            end = 1000000
        else:
            end = 5000
        self.filenames = self.filenames[:end]
        
        print(os.name, len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        Get landmark alignment outside in train_pass()
        """

        line = self.filenames[item]
        p, s_idx, e_idx = line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2])

        imgs_path = glob.glob(p + '/*.jpg')
        imgs_path.sort()

        frames = []

        for j in range(s_idx, e_idx, 3):
            #(256, 256, 3)
            face_img = [pil_loader(imgs_path[j + m]) for m in range(-2, 3)]
            #(3， 224，224)
            face_img = self.transform(face_img) 
            #(6, 256, 256, 3)
            face_img = np.stack(face_img, axis=0)
            #(3, ,6, 224, 224)
            frames.append(face_img.transpose((1, 0, 2, 3)))
            
        frames = np.stack(frames, axis=0).astype(np.float32)  # seq_len x 256 x 256 x 3

         # audio
        audio_path = os.path.join(p.replace('raw_frame', 'raw_wav') + '.wav')
        samples, sample_rate = sf.read(audio_path)
        assert sample_rate == 16000

        D = pySTFT(samples).T
        mel_basis = mel(16000, 320, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)  

        audio_in = []
        for item in range(s_idx, e_idx, 3):
            sel_audio_clip = S[(item - 2) * 4:(item + 3) * 4, :]
            for _ in range(20 - sel_audio_clip.shape[0]):
                sel_audio_clip = np.concatenate((sel_audio_clip, sel_audio_clip[-1:]), axis=0)
                print('*****************************')
            assert sel_audio_clip.shape[0] == 20
            audio_in.append(sel_audio_clip)

        audio_in = np.stack(audio_in, axis=0).astype(np.float32)

        assert audio_in.shape[0] == frames.shape[0]
        return audio_in, frames

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)
    

