import torch.utils.data as data
import numpy as np
import os
from medpy.io import load
import pandas as pd

class SourceDataset(data.Dataset):
    def __init__(self, src_data_dir, mode, num_tg_data, fold_number=0, total_fold=10):
        self.src_data_dir = src_data_dir     ### /mnt/nas2/data/lung_nodule_sequence/
        self.src_seq_vol_dir = self.src_data_dir + 'seq_scan/'
        self.src_seq_mask_dir = self.src_data_dir + 'seq_mask/'
        
        src_patient_ids = np.array(pd.read_csv(self.src_data_dir + 'patient_id.csv').patient_id)
        src_patient_ids = sorted(src_patient_ids)
        
        src_test_patient_ids = src_patient_ids[fold_number::total_fold]
        if mode == "train":
            self.src_mode_patient_ids = [i for i in src_patient_ids if i not in src_test_patient_ids]
        elif mode == "test":
            self.src_mode_patient_ids = src_test_patient_ids
        ###
        self.src_mode_patient_ids = np.random.choice(self.src_mode_patient_ids, num_tg_data)

        self.src_seq_filename = self.load_filenames(self.src_seq_vol_dir, self.src_mode_patient_ids)
        
        # print(self.src_seq_filename)
        # print(len(self.src_mode_patient_ids))
        # print(len(self.src_seq_filename))

    def load_filenames(self, data_dir, mode_patient_id):
        target_filenames = []

        filenames = os.listdir(data_dir)
        for filename in filenames:
            patient_id = filename.split("_")[0]
            if patient_id in mode_patient_id:
                target_filenames.append(filename)
                
        return target_filenames

    def get_volume(self, data_dir):
        volume, _ = load(data_dir)
        volume = np.expand_dims(volume, axis=0)
        volume = np.transpose(volume, (3, 0, 1, 2))        
        return volume    ### [sequence, channel, x, y]
    
    def get_mask(self, data_dir):
        mask, _ = load(data_dir)
        mask = mask[:,:,1]
        mask = np.expand_dims(mask, axis = 0)   
        return mask      ### [channel, x, y]

    def __len__(self):
        return len(self.src_seq_filename)
    
    def __getitem__(self, index):
        vol_path = self.src_seq_vol_dir + self.src_seq_filename[index]
        mask_path = self.src_seq_mask_dir + self.src_seq_filename[index]

        vol = self.get_volume(vol_path)
        mask = self.get_mask(mask_path)

        return vol, mask