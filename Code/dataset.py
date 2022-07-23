"""
Training script for the unconditional samplernn Pytorch implementation
Date :  27 Dec 2021
Updated :  29 Dec 2021
Written By :  Mukul Agarwal 62380
"""

import os
import torch
import utils
import random
import fnmatch
import warnings
import numpy as np
import logging
from os import listdir
from os.path import join
from natsort import natsorted
import librosa
from librosa.core import load
from torch.utils.data import (Dataset, DataLoader as DataLoaderBase)

# We need an inital random shuffle, which remain same across the runs, even if we
# resume the training, much better idea is to do a random.Random instance.
def round_to(x, base=5):
    return base * round(x/base)

def truncate_to(x, base):
    return int(np.floor(x / float(base))) * base

def find_files(directory, pattern='*.wav'):
    files = []
    for root , dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root,filename))
    return files


def get_dataset_filenames_split(data_dir, val_frac, batch_size):
    files = find_files(data_dir)
    assert batch_size <= len(files), 'Batch size exceeds the corpus length'
    if not files:
        raise ValueError(f'No wan files found in {data_dir}.')
    random.Random(4).shuffle(files)

    if not (len(files) % batch_size) == 0:
        warnings.warn('Truncating dataset, length is not equally divisible by batch size')
        idx = truncate_to(len(files), batch_size)
        files = files[: idx]
    val_size = len(files)*val_frac
    val_size = round_to(val_size, batch_size)
    if val_size == 0 : val_size = batch_size
    val_start = len(files) - val_size
    return files[: val_start], files[val_start :]


def get_dataset(split,epochs,batch_size,seq_len,overlap,drop_remainder,q_type,q_levels):
    pass

class FolderDataset(Dataset):

    def __init__(self, path,frame_len,overlap_len,ratio_min=0, ratio_max=1):
        super().__init__()
        self.frame_len = frame_len
        self.overlap_len = self.frame_len//2
        file_names = natsorted([join(path, file_name) for file_name in listdir(path)])
        random.Random(4).shuffle(file_names)
        self.file_names = file_names[int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))]

    def __getitem__(self, index):
        (seq,_) = load(self.file_names[index], sr=None, mono=True)
        # logging.info("File Loaded : ", self.file_names[index])
        print("File Loaded : ", self.file_names[index])
        # # dataa = utils.mu_law_quantization(torch.from_numpy(seq), self.q_levels)
        # cat_data = torch.cat([
        #     torch.LongTensor(self.overlap_len).fill_(utils.q_zero(self.q_levels)),
        #     utils.mu_law_quantization(torch.from_numpy(seq),self.q_levels)
        # ])
        #
        # return cat_data[:(cat_data.shape[0]//self.seq_len)*self.seq_len + self.overlap_len]
        # return dataa
        return seq[:((len(seq)-self.frame_len)//self.overlap_len+1)*self.overlap_len]
        # return seq[:((len(seq))//self.frame_len) * self.frame_len]

    def __len__(self):
        return len(self.file_names)


class Dataloader(DataLoaderBase):

    def __init__(self, dataset, batch_size, overlap_len,seq_len,*args, **kwargs):
        super().__init__(dataset, batch_size,*args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = self.seq_len//2


    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()
            batch_reset = True
            hidden_reset = True


            # input_sequeces = batch[:,:]
            # target_sequences = batch[:,:]
            # yield (input_sequeces, hidden_reset,batch_reset, target_sequences)


            for seq_begin in range(0, n_samples, self.overlap_len):
                from_index = seq_begin
                to_index = seq_begin + self.seq_len

                sequences = batch[:, from_index: to_index]
                if sequences.shape[1] != self.seq_len:
                    zeros = torch.zeros(batch_size,self.seq_len-sequences.shape[1])
                    sequences = torch.cat((sequences,zeros), dim=1)
                input_sequeces = sequences[:,:]
                target_sequences = sequences[:, :]


                yield (input_sequeces,hidden_reset,batch_reset,target_sequences)

                hidden_reset = False
                batch_reset = False

                # for seq_begin in range(0, n_samples, self.seq_len):
                #     from_index = seq_begin
                #     to_index = seq_begin + self.seq_len
                #
                #     sequences = batch[:, from_index: to_index]
                #     input_sequeces = sequences[:, :]
                #     target_sequences = sequences[:, :]
                #
                #     yield (input_sequeces, hidden_reset, batch_reset, target_sequences)
                #
                #     hidden_reset = False
                #     batch_reset = False


    def __len__(self):
        raise NotImplementedError()

# "Checking the implementation"
# data_dir = "chunks"
# train_data, val_data = get_dataset_filenames_split(data_dir, 0.1, 4)
# print("YO")