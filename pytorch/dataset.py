import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# Simple PyTorch Dataset wrapper for pre-processed data

train_file="NASA_Turbofan_Engine_Degradation/train_FD003.txt"
test_file="NASA_Turbofan_Engine_Degradation/test_FD003.txt"
truth_file="NASA_Turbofan_Engine_Degradation/RUL_FD003.txt"

class PreparedCMAPSSDataset(Dataset):
    def __init__(self, features, labels):
        self.x_data = torch.tensor(features, dtype=torch.float32)
        self.y_data = torch.tensor(labels, dtype=torch.float32) # Float32 for BCEWithLogitsLoss
        
    def __len__(self):
        return len(self.x_data)
        
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def prepare_cmapss_data(train_file=train_file, test_file=test_file, truth_file=truth_file, rul_threshold=30):
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

    # --- 1. PROCESS TRAINING DATA ---
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)
    train_max_cycles = train_df.groupby('unit_nr')['time_cycles'].transform('max')
    train_df['RUL'] = train_max_cycles - train_df['time_cycles']
    train_df['label'] = (train_df['RUL'] <= rul_threshold).astype(int)

    # --- 2. PROCESS TEST DATA ---
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
    truth_df = pd.read_csv(truth_file, sep=r'\s+', header=None, names=['True_RUL'])
    truth_df['unit_nr'] = truth_df.index + 1  # Engines are 1-indexed

    # Calculate absolute failure cycle: (Last recorded test cycle) + (True remaining RUL)
    test_max_cycles = test_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    test_max_cycles = test_max_cycles.merge(truth_df, on='unit_nr')
    test_max_cycles['absolute_failure_cycle'] = test_max_cycles['time_cycles'] + test_max_cycles['True_RUL']

    # Map back to test_df and calculate RUL for every row
    test_df = test_df.merge(test_max_cycles[['unit_nr', 'absolute_failure_cycle']], on='unit_nr')
    test_df['RUL'] = test_df['absolute_failure_cycle'] - test_df['time_cycles']
    test_df['label'] = (test_df['RUL'] <= rul_threshold).astype(int)

    # --- 3. FEATURE NORMALIZATION ---
    scaler = StandardScaler()
    
    # FIT the scaler ONLY on training data, then transform both
    train_features = scaler.fit_transform(train_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])

    # --- 4. CREATE PYTORCH DATASETS ---
    train_dataset = PreparedCMAPSSDataset(train_features, train_df['label'].values)
    test_dataset = PreparedCMAPSSDataset(test_features, test_df['label'].values)

    return train_dataset, test_dataset, scaler