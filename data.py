from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
from sklearn.preprocessing import MinMaxScaler



"""
class SleepDataset(Dataset):
    def __init__(self, folder, CFG, is_train=False, zbp=0.1, sample_per_epoch=10000):
        self.anglez_mean = np.load('anglez_mean.npy')
        self.anglez_std = np.load('anglez_std.npy')
        self.enmo_mean = np.load('enmo_mean.npy')
        self.enmo_std = np.load('enmo_std.npy')
        
        self.CFG = CFG
        self.is_train = is_train
        self.sample_per_epoch = sample_per_epoch
        self.feat_list = ['anglez', 'enmo', 'hour'] #'step',
        self.label_list = ['asleep']
        #self.Xys = self.read_csvs(folder)
        self.slide_step = CFG['block_stride']  # Sliding window step size based on CFG
        #self.total_full_blocks = sum((len(df) - self.CFG['block_size']) // self.slide_step + 1 for df in self.Xys)

        self.Xys = self.read_csvs(folder)
        self.Xys, non_zero_count, kept_zero_count = self.filter_non_zero_blocks(self.Xys, zbp)
        self.total_full_blocks = non_zero_count + kept_zero_count


    def read_csvs(self, folder):
        res = []
        if isinstance(folder, str):
            files = glob.glob(f'{folder}/*.csv')
        else:
            files = folder
        for i, f in enumerate(files):
            df = pd.read_parquet(f)
            df = self.norm_feat_eng(df)
            res.append(df)
        return res

    def filter_non_zero_blocks(self, Xys, zero_block_percentage=0.1):
        zero_blocks = []
        non_zero_blocks = []

        # Separate zero and non-zero blocks
        for df in Xys:
            for start_idx in range(0, len(df), self.slide_step):
                end_idx = min(start_idx + self.CFG['block_size'], len(df))
                block = df[start_idx:end_idx]
                y_block = block[self.label_list].values

                if np.all(y_block == 0):
                    zero_blocks.append(block)
                else:
                    non_zero_blocks.append(block)

        # Randomly select a percentage of zero blocks to keep
        num_zero_blocks_to_keep = int(len(zero_blocks) * zero_block_percentage)
        kept_zero_blocks = np.random.choice(zero_blocks, num_zero_blocks_to_keep, replace=False)

        # Combine non-zero blocks and kept zero blocks
        filtered_Xys = non_zero_blocks + list(kept_zero_blocks)
        print(len(non_zero_blocks), len(kept_zero_blocks))

        return filtered_Xys, len(non_zero_blocks), len(kept_zero_blocks)




    def norm_feat_eng(self, X):
            #X['anglez'] = (X['anglez'] - self.anglez_mean) / (self.anglez_std + 1e-12)
            #X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-12)

            X['hour'] = X['hour'] / 24  # Normalize hour feature

            # Remove the 'step' column
            if 'step' in X.columns:
                X.drop(columns=['step'], inplace=True)

            return X.fillna(0).astype(np.float32)


    def __len__(self):
        return self.total_full_blocks
        #return sum((len(df) - self.CFG['block_size']) // self.slide_step + 1 for df in self.Xys)



    def __getitem__(self, index):
        if index >= len(self.Xys):
            raise IndexError(f"Index {index} out of range. Total blocks: {len(self.Xys)}")

        Xy = self.Xys[index]

        # Extract features and labels
        X = Xy[self.feat_list].values.astype(np.float32)
        y = Xy[self.label_list].values.astype(np.float32)

        # Apply thresholding to the targets
        threshold = 0.9  # Define your threshold here
        y = (y > threshold).astype(np.float32)

        # Add noise to 50% of the data
        if random.random() < 0.1:  # 50% chance
            # Generate noise for 'enmo' and 'anglez' features based on their respective stds
            noise_enmo = np.random.normal(0, self.enmo_std, X[:, 1].shape)
            noise_anglez = np.random.normal(0, self.anglez_std, X[:, 0].shape)

            X[:, 1] += noise_enmo
            X[:, 0] += noise_anglez

            # Perturb 'hour' feature within a sensible range, e.g., +/- 1 hour
            time_shift = np.random.uniform(-1/24, 1/24)
            X[:, 2] = (X[:, 2] + time_shift) % 1  # Ensure hour stays within 0-24 range

        # Further process the segment if needed
        X_processed, y_processed = self.process_data(X, y)

        # Return the processed segment
        return {"X": X_processed}, {"Y": y_processed}


    def process_data(self, X, y):
        # Ensure block_size is divisible by patch_size
        required_length = self.CFG['block_size'] // self.CFG['patch_size'] * self.CFG['patch_size']
        #print("Required length:", required_length)
        #print("Original X shape:", X.shape)

        blocks_X = []
        blocks_y = []

        # Break data into blocks
        for start_idx in range(0, len(X), required_length):
            end_idx = min(start_idx + required_length, len(X))
            block_X = X[start_idx:end_idx]
            block_y = y[start_idx:end_idx]
            #print("Block X shape:", block_X.shape)

            # Adjust block length if necessary
            if len(block_X) < required_length:
                pad_size = required_length - len(block_X)
                block_X = np.pad(block_X, ((0, pad_size), (0, 0)))
                block_y = np.pad(block_y, ((0, pad_size), (0, 0)))
                #print("Block X shape after padding:", block_X.shape)

            # Reshape to patches
            block_X = torch.tensor(block_X).view(-1, self.CFG['patch_size'], block_X.shape[1])
            #print("Block X shape after reshaping to patches:", block_X.shape)
            block_y = torch.tensor(block_y).view(-1, self.CFG['patch_size'], block_y.shape[1])

            # Flatten patches for X
            block_X = block_X.reshape(-1, self.CFG['patch_size'] * block_X.shape[2])

            # Max Pooling across patches for y
            block_y, _ = torch.max(block_y, dim=1)

            blocks_X.append(block_X)
            blocks_y.append(block_y)

        # Stack all blocks
        X_processed = torch.cat(blocks_X, dim=0)

        y_processed = torch.cat(blocks_y, dim=0)

        return X_processed, y_processed
"""







class SleepDataset(Dataset):
    def __init__(self, folder, CFG, is_train=False, zbp=0.1, target_thresh=None):
        self.anglez_mean = np.load('anglez_mean.npy')
        self.anglez_std = np.load('anglez_std.npy')
        self.enmo_mean = np.load('enmo_mean.npy')
        self.enmo_std = np.load('enmo_std.npy')

        self.CFG = CFG
        self.is_train = is_train
        self.feat_list = ['anglez', 'enmo']#, 'anglez_enmo', 'anglez_fft_feature'] # 'hour', 'step',
        self.label_list = ['onset', 'wakeup']
        #self.Xys = self.read_csvs(folder)
        self.slide_step = CFG['block_stride']  # Sliding window step size based on CFG
        #self.total_full_blocks = sum((len(df) - self.CFG['block_size']) // self.slide_step + 1 for df in self.Xys)
        
        self.Xys = self.read_csvs(folder)
        self.Xys, non_zero_count, kept_zero_count = self.filter_non_zero_blocks(self.Xys, zbp)
        self.total_full_blocks = non_zero_count + kept_zero_count

        self.target_thresh= target_thresh

    def read_csvs(self, folder):
        res = []
        if isinstance(folder, str):
            files = glob.glob(f'{folder}/*.parquet')
        else:
            files = folder
        for i, f in enumerate(files):
            df = pd.read_parquet(f)
            df = self.norm_feat_eng(df)
            res.append(df)
        return res
    
    def filter_non_zero_blocks(self, Xys, zero_block_percentage=0.1):
        zero_blocks = []
        non_zero_blocks = []

        # Separate zero and non-zero blocks
        for df in Xys:
            for start_idx in range(0, len(df), self.slide_step):
                end_idx = min(start_idx + self.CFG['block_size'], len(df))
                block = df[start_idx:end_idx]
                y_block = block[self.label_list].values

                if np.all(y_block == 0):
                    zero_blocks.append(block)
                else:
                    non_zero_blocks.append(block)

        # Randomly select a percentage of zero blocks to keep
        num_zero_blocks_to_keep = int(len(zero_blocks) * zero_block_percentage)
        kept_zero_blocks = np.random.choice(zero_blocks, num_zero_blocks_to_keep, replace=False)

        # Combine non-zero blocks and kept zero blocks
        filtered_Xys = non_zero_blocks + list(kept_zero_blocks)
        print(len(non_zero_blocks), len(kept_zero_blocks))

        return filtered_Xys, len(non_zero_blocks), len(kept_zero_blocks)

    
    
    def norm_feat_eng(self, X):
            #X['anglez'] = (X['anglez'] - self.anglez_mean) / (self.anglez_std + 1e-12)
            #X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-12)

            X['hour'] = X['hour'] / 24  # Normalize hour feature


            X = X[['anglez', 'enmo', 'onset', 'wakeup']]

            return X.fillna(0).astype(np.float32)
    """
    def norm_feat_eng(self, X):
        # Normalize the existing 'anglez' and 'enmo' features if needed
        # X['anglez'] = (X['anglez'] - self.anglez_mean) / (self.anglez_std + 1e-12)
        # X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-12)
    
        # Calculate anglez in radians and its cosine
        X['anglez_radians'] = (np.pi / 180) * X['anglez']
        X['cos_anglez'] = np.cos(X['anglez_radians'])
    
        # Clip the 'enmo' feature between 0 and 1
        X['enmo'] = np.clip(X['enmo'], 0, 1)
    
        # Normalize hour feature
        #X['hour'] = X['hour'] / 24
    
        # Select the features to include in the model
        X = X[['anglez', 'enmo', 'cos_anglez', 'onset', 'wakeup']]
    
        # Fill missing values with 0 and ensure the type is float32
        return X.fillna(0).astype(np.float32)"""
    


    """def norm_feat_eng(self, X):
        # Add a cross feature: the product of anglez and enmo
        X['anglez_enmo'] = X['anglez'] * X['enmo']
    
        # Apply FFT to anglez and get absolute values to include as features
        anglez_fft = np.fft.fft(X['anglez'].values)
        anglez_fft_abs = np.abs(anglez_fft)
        
        # Since FFT output is symmetrical, we can take the first coefficient
        # after the zero frequency term as our feature.
        X['anglez_fft_feature'] = anglez_fft_abs  # Index 1 is the first non-zero frequency term

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        anglez_fft_feature = X['anglez_fft_feature'].values.reshape(-1, 1)
        X['anglez_fft_feature'] = scaler.fit_transform(anglez_fft_feature).flatten()

        X.fillna(0, inplace=True)
    
        # Normalize anglez and enmo using z-score normalization
        # You can uncomment these lines if you wish to apply z-score normalization
        # X['anglez'] = (X['anglez'] - self.anglez_mean) / (self.anglez_std + 1e-12)
        # X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-12)
    
        # Select the desired columns for the output dataframe
        X = X[['anglez', 'enmo', 'anglez_enmo', 'anglez_fft_feature', 'onset', 'wakeup']]
    
        return X"""

        

    def __len__(self):
        return self.total_full_blocks
        #return sum((len(df) - self.CFG['block_size']) // self.slide_step + 1 for df in self.Xys)
        

    def __getitem__(self, index):
        if index >= len(self.Xys):
            raise IndexError(f"Index {index} out of range. Total blocks: {len(self.Xys)}")

        Xy = self.Xys[index]

        # Extract features and labels
        X = Xy[self.feat_list].values.astype(np.float32)
        y_onset = Xy['onset'].values.astype(np.float32)
        y_wakeup = Xy['wakeup'].values.astype(np.float32)

        y = np.stack([y_onset, y_wakeup], axis=1)  # Combine the targets

        # Further process the segment if needed
        X_processed, y_processed = self.process_data(X, y)

        return {"X": X_processed}, {"Y": y_processed}


        
    def process_data(self, X, y):
        # Ensure block_size is divisible by patch_size
        required_length = self.CFG['block_size'] // self.CFG['patch_size'] * self.CFG['patch_size']
        #print("Required length:", required_length)
        #print("Original X shape:", X.shape)

        blocks_X = []
        blocks_y = []

        # Break data into blocks
        for start_idx in range(0, len(X), required_length):
            end_idx = min(start_idx + required_length, len(X))
            block_X = X[start_idx:end_idx]
            block_y = y[start_idx:end_idx]
            #print("Block X shape:", block_X.shape)

            # Adjust block length if necessary
            if len(block_X) < required_length:
                pad_size = required_length - len(block_X)
                block_X = np.pad(block_X, ((0, pad_size), (0, 0)))
                block_y = np.pad(block_y, ((0, pad_size), (0, 0)))
                #print("Block X shape after padding:", block_X.shape)

            # Reshape to patches
            block_X = torch.tensor(block_X).view(-1, self.CFG['patch_size'], block_X.shape[1])
            #print("Block X shape after reshaping to patches:", block_X.shape)
            block_y = torch.tensor(block_y).view(-1, self.CFG['patch_size'], block_y.shape[1])

            # Flatten patches for X
            block_X = block_X.reshape(-1, self.CFG['patch_size'] * block_X.shape[2])

            # Max Pooling across patches for y
            block_y, _ = torch.max(block_y, dim=1)

            blocks_X.append(block_X)
            blocks_y.append(block_y)

        # Stack all blocks
        X_processed = torch.cat(blocks_X, dim=0)
        
        y_processed = torch.cat(blocks_y, dim=0)

        return X_processed, y_processed


