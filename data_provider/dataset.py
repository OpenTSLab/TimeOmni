import os
import warnings
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not available. .fif files will not be supported.")


class Dataset_Unified(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, jsonl_file_path=None, base_dir=None, use_scaler=True, unfold_channels=True, items: list=None, max_tokens=2000):
        """
        Unified dataset class that loads time-series data in the dataset.
        :param jsonl_file_path: JSONL file path
        :param tokenizer: Hugging Face tokenizer
        :param base_dir: root directory for media files (optional)
        :param use_scaler: whether to standardize time-series data
        :param unfold_channels: whether to reshape output from (T, C) to (T*C, 1)
        :param items: pass data list directly, same format as JSONL
        :param max_tokens: max text token length (truncate if exceeded)
        """
        self.data = []
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self.use_scaler = use_scaler
        self.unfold_channels = unfold_channels
        self.max_tokens = max_tokens

        assert jsonl_file_path or items, "Either jsonl_file_path or items must be provided"
        assert jsonl_file_path is None or items is None, "Provide only one of jsonl_file_path or items"
        
        if jsonl_file_path:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.data.append(sample)
        elif items:
            self.data = items

    def load_ts_data(self, file_path: str, ori_file_path: str = None) -> tuple:
        """
        Load time-series data by file extension. Returns T x C (time x channels) or T*C x 1.

        :param file_path: data file path
        :param ori_file_path: original file path before segmentation
        :return: tuple (data_tensor, mask_tensor)
            - data_tensor: tensor shaped (T, C) or (T*C, 1)
            - mask_tensor: missing-value mask, same shape as data_tensor; 'X' positions are 1
            - ori_channels: original channel count
        """

        def _load_ts_data(file_path: str) -> tuple:
            """
            Internal helper to load time-series data by file extension.
            Supported formats:
            - Audio: .wav, .mp3, .flac, .m4a (via torchaudio)
            - CSV: .csv (each column is a channel)
            - NumPy: .npy
            - MNE: .fif (EEG/MEG data)

            Returns:
                tuple: (np_data, missing_mask)
                    - np_data: time-series data
                    - missing_mask: 'X' missing-value mask, same shape as np_data
            """
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            missing_mask = None
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.wav', '.mp3', '.flac', '.m4a']:
                # Audio file: use torchaudio.
                waveform, _ = torchaudio.load(file_path)
                # torchaudio returns (C, T), transpose to (T, C).
                np_data = waveform.transpose(0, 1).numpy()
                
            elif file_ext == '.csv':
                # CSV file: assume each column is a channel.
                df = pd.read_csv(file_path, header=None)
                
                # Detect 'X' missing values and build mask.
                x_mask = (df == 'X') | (df == 'x')
                
                if x_mask.any().any():
                    missing_mask = x_mask.astype(np.float32).values  # Convert to numpy array.
                    # Replace 'X' with NaN.
                    df = df.replace(['X', 'x'], np.nan)
                    # Convert to numeric.
                    df = df.apply(pd.to_numeric, errors='coerce')
                    # Fill missing values via linear interpolation.
                    df = df.interpolate(method='linear', limit_direction='both')
                
                np_data = df.values
                
            elif file_ext == '.npy':
                # NumPy file.
                np_data = np.load(file_path)
                # If 1D, expand to (T, 1).
                if np_data.ndim == 1:
                    np_data = np_data.reshape(-1, 1)
                    
            elif file_ext == '.fif':
                # MNE file (EEG/MEG data).
                if not MNE_AVAILABLE:
                    raise ImportError("MNE-Python is required to read .fif files. Please install it with: pip install mne")
                
                # Read raw data.
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
                # MNE returns (n_channels, n_times), transpose it.
                np_data = raw.get_data().T  # Transpose to (T, C).
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            return np_data, missing_mask

        try:
            np_data, missing_mask = _load_ts_data(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a small placeholder tensor to avoid crashing.
            return torch.zeros((1, 1), dtype=torch.float32), None

        # Max length: 240k.
        max_length = 240000
        np_data = np_data[:max_length]
        # Use SimpleImputer to fill NaN and inf.
        if np_data.size > 0:
            # Replace inf/-inf with NaN first.
            np_data = np.where(np.isfinite(np_data), np_data, np.nan)
            # If all NaN, return placeholder.
            if np.isnan(np_data).all():
                return torch.zeros((1, 1), dtype=torch.float32), None
            # Fill NaN with column means.
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            np_data = imputer.fit_transform(np_data)
        missing_mask = missing_mask[:max_length] if missing_mask is not None else None

        assert np_data.ndim == 2, f"Data must be 2D array with shape (T, C), got shape {np_data.shape}"
        
        # Validate data type.
        if not np.issubdtype(np_data.dtype, np.number):
            raise ValueError(f"Data contains non-numeric values. Data type: {np_data.dtype}, sample data: {np_data[:5]}")

        # Standardize data if enabled.
        if self.use_scaler:
            # Create a scaler per sample to avoid multi-process conflicts.
            local_scaler = StandardScaler()
            if ori_file_path and os.path.exists(ori_file_path):
                # If original path is provided, fit on original data.
                ori_np_data, ori_missing_mask = _load_ts_data(ori_file_path)
                if ori_np_data.ndim == 2:
                    local_scaler.fit(ori_np_data)
                    np_data = local_scaler.transform(np_data)
                else:
                    raise ValueError(f"Original data must be 2D array with shape (T, C), got shape {ori_np_data.shape}")
            else:
                np_data = local_scaler.fit_transform(np_data)

        # Convert to float32 tensors.
        data = torch.from_numpy(np_data.astype(np.float32))
        mask = torch.from_numpy(missing_mask.astype(np.float32)) if missing_mask is not None else None

        # Flatten based on unfold_channels.
        if self.unfold_channels:
            data = data.reshape(-1, 1)  # Flatten directly to (T*C, 1).
            mask = mask.reshape(-1, 1) if mask is not None else None   # Flatten mask to (T*C, 1).
        
        return data, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_text = random.choice(sample['input_text'])
        gt_text = random.choice(sample.get("gt_text", [""]))
        input_text_chat = {"role": "user", "content": input_text}

        if "gt_ts" in sample and sample["gt_ts"]:
            # Forecast task: input only, no target text.
            input_ids = self.tokenizer.apply_chat_template(
                [input_text_chat],
                tokenize=True,
                truncation=True,
                max_length=self.max_tokens
            )
            labels = None  # No labels needed for forecasting.
        else:
            target_text = {"role": "assistant", "content": gt_text}
            # Tokenize
            input_encoding = self.tokenizer.apply_chat_template(
                [input_text_chat],
                tokenize=True,
                truncation=True,
                max_length=self.max_tokens
            )
            target_encoding = self.tokenizer.apply_chat_template(
                [target_text],
                tokenize=True,
                truncation=True,
                max_length=self.max_tokens
            )
            # Concatenate input + target for causal LM training.
            input_ids = input_encoding + target_encoding
            labels = [-100] * len(input_encoding) + target_encoding  # Compute loss on assistant part only.
            labels[-1] = -100 # Do not compute loss for the last token.

        input_ts = sample['input_ts']
        if input_ts:
            if input_ts['already_segment']:
                input_ts_path = input_ts['segment']['seg_path']
            else:
                input_ts_path = input_ts['original']['ori_path']
            if self.base_dir:
                input_ts_path = os.path.join(self.base_dir, input_ts_path)
        
            input_ts_tensor = None
            if input_ts_path:
                if input_ts['already_segment']:
                    input_ts_tensor, input_ts_mask = self.load_ts_data(input_ts_path)
                else:
                    input_ts_tensor, input_ts_mask = self.load_ts_data(input_ts_path)  # Return data and mask.
        else:
            input_ts_path = None
            input_ts_tensor = torch.ones((1, 1), dtype=torch.float32) * 10 # Placeholder to avoid None.
            input_ts_mask = None

        result = {
            "id": sample["id"],
            "dataset_name": sample.get("dataset_name", ""),
            "task": sample.get("task", ""),
            "scene": sample.get("scene", ""),
            "uid": sample.get("uid", ""),
            "input_ids": input_ids,
            "labels": labels,
            "input_text": input_text,
            "gt_text": gt_text,
            "gt_result": sample.get("gt_result", None),
            "input_ts": input_ts_tensor,
            "input_ts_mask": input_ts_mask,
            "input_ts_path": input_ts_path
        }

        # If target path exists, also load target time series.
        if "gt_ts" in sample and sample["gt_ts"]:
            gt_ts_path = sample["gt_ts"]["path"]
            if self.base_dir:
                gt_ts_path = os.path.join(self.base_dir, gt_ts_path)
            
            gt_ts = None
            gt_ts_mask = None
            if gt_ts_path:
                gt_ts, gt_ts_mask = self.load_ts_data(gt_ts_path)  # Return tensor and mask.
            
            result["gt_ts"] = gt_ts
            result["gt_ts_mask"] = gt_ts_mask
            result["gt_ts_path"] = gt_ts_path

        return result

@dataclass
class Collator_Unified:
    tokenizer: AutoTokenizer
    # Control padding side for time (sequence) and channel dimensions. Options: 'right' (default) or 'left'
    pad_side_time: str = "right"
    pad_side_channel: str = "right"

    def _pad_ts_tensors(self, ts_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad a list of time-series tensors to unify channels and time length.

        :param ts_list: list of tensors shaped (T, C)
        :return: padded tensor shaped (Batch, Max_Time, Max_Channels)
        """
        if not ts_list or any(ts is None for ts in ts_list):
            return None
            
        # Compute max sizes.
        max_channels = max(ts.shape[1] for ts in ts_list if ts is not None)
        max_time = max(ts.shape[0] for ts in ts_list if ts is not None)

        padded_ts_list = []
        for ts in ts_list:
            # Pad channels to max_channels.
            T, C = ts.shape
            if C < max_channels:
                add_c = max_channels - C
                pad_c = torch.zeros(T, add_c, dtype=ts.dtype, device=ts.device)
                if self.pad_side_channel == 'left':
                    ts = torch.cat([pad_c, ts], dim=1)
                else:  # default: right
                    ts = torch.cat([ts, pad_c], dim=1)

            # Pad time to max_time.
            T = ts.shape[0]
            if T < max_time:
                add_t = max_time - T
                pad_t = torch.zeros(add_t, ts.shape[1], dtype=ts.dtype, device=ts.device)
                if self.pad_side_time == 'left':
                    ts = torch.cat([pad_t, ts], dim=0)
                else:  # default: right
                    ts = torch.cat([ts, pad_t], dim=0)

            padded_ts_list.append(ts)

        # Stack into batch tensor: (Batch, Max_Time, Max_Channels).
        return torch.stack(padded_ts_list, dim=0)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        input_ts_list = [item["input_ts"] for item in batch]
        input_ts_mask_list = [item.get("input_ts_mask", None) for item in batch]
        input_ts_paths = [item["input_ts_path"] for item in batch]
        ids = [item["id"] for item in batch]
        uids = [item.get("uid", "") for item in batch]
        dataset_names = [item.get("dataset_name", "") for item in batch]
        tasks = [item.get("task", "") for item in batch]
        scenes = [item.get("scene", "") for item in batch]
        input_texts = [item["input_text"] for item in batch]
        gt_texts = [item["gt_text"] for item in batch]
        gt_results = [item.get("gt_result", None) for item in batch]

        # Dynamic padding - handle input_ids=None.
        input_ids_tensor = None
        if None not in input_ids:
            input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
        
        # Handle labels; skip padding if None present.
        labels_tensor = None
        if None not in labels:
            labels_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(lbls, dtype=torch.long) for lbls in labels],
                batch_first=True,
                padding_value=-100
            )

        result = {
            "input_ids": input_ids_tensor,  # May be None.
            "labels": labels_tensor,  # May be None.
            "input_ts_paths": input_ts_paths,
            "input_ts_list": input_ts_list,
            "input_ts_mask_list": input_ts_mask_list,
            "ids": ids,
            "uids": uids,
            "dataset_names": dataset_names,
            "tasks": tasks,
            "scenes": scenes,
            "input_texts": input_texts,
            "gt_texts": gt_texts,
            "gt_results": gt_results,
        }

        # If target time-series data exists, process it as well.
        if "gt_ts" in batch[0]:
            gt_ts_list = [item["gt_ts"] for item in batch]
            result['gt_ts_list'] = gt_ts_list
            result["gt_ts_paths"] = [item.get("gt_ts_path") for item in batch]
            result["gt_ts_mask_list"] = [item.get("gt_ts_mask", None) for item in batch]

        return result
