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
        统一的数据集类，直接在dataset中加载音频
        :param jsonl_file_path: JSONL 文件路径
        :param tokenizer: Hugging Face tokenizer
        :param base_dir: 音频文件根目录（可选）
        :param use_scaler: 是否对时间序列数据进行标准化处理
        :param unfold_channels: 是否将输出从 (T, C) 变换为 (T*C, 1)，默认为True
        :param items: 直接传入数据列表，格式同jsonl文件内容
        :param max_tokens: 最大文本token数，超过则截断
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
        根据文件后缀读取时间序列数据，返回格式为 T x C (时间 x 通道) 或 T*C x 1
        
        :param file_path: 数据文件路径
        :param ori_file_path: 切分之前的原始文件路径
        :return: 元组 (data_tensor, mask_tensor)
                - data_tensor: 数据张量，形状为 (T, C) 或 (T*C, 1)
                - mask_tensor: 缺失值mask，形状与data_tensor相同，'X'位置为1，其他为0
                - ori_channels: 原始通道数
        """

        def _load_ts_data(file_path: str) -> tuple:
            """
            内部函数，根据文件后缀读取时间序列数据
            支持格式：
            - 音频文件：.wav, .mp3, .flac, .m4a (使用 torchaudio)
            - CSV文件：.csv (假设每列是一个通道)
            - NumPy文件：.npy
            - MNE文件：.fif (EEG/MEG数据)
            
            Returns:
                tuple: (np_data, missing_mask) 
                    - np_data: 时间序列数据
                    - missing_mask: 'X'缺失值的mask，形状与np_data相同，'X'位置为1，其他为0
            """
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            missing_mask = None
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.wav', '.mp3', '.flac', '.m4a']:
                # 音频文件：使用torchaudio
                waveform, _ = torchaudio.load(file_path)
                # torchaudio返回格式为 (C, T)，需要转置为 (T, C)
                np_data = waveform.transpose(0, 1).numpy()
                
            elif file_ext == '.csv':
                # CSV文件：假设每列是一个通道
                df = pd.read_csv(file_path, header=None)
                
                # 检测'X'缺失值并创建mask
                x_mask = (df == 'X') | (df == 'x')
                
                if x_mask.any().any():
                    missing_mask = x_mask.astype(np.float32).values  # 转换为numpy数组
                    # 将'X'替换为NaN
                    df = df.replace(['X', 'x'], np.nan)
                    # 转换为数值类型
                    df = df.apply(pd.to_numeric, errors='coerce')
                    # 使用线性插值填补缺失值
                    df = df.interpolate(method='linear', limit_direction='both')
                
                np_data = df.values
                
            elif file_ext == '.npy':
                # NumPy文件
                np_data = np.load(file_path)
                # 如果是1D数组，扩展为 (T, 1)
                if np_data.ndim == 1:
                    np_data = np_data.reshape(-1, 1)
                    
            elif file_ext == '.fif':
                # MNE文件（EEG/MEG数据）
                if not MNE_AVAILABLE:
                    raise ImportError("MNE-Python is required to read .fif files. Please install it with: pip install mne")
                
                # 读取raw数据
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
                # 获取数据：MNE返回 (n_channels, n_times)，需要转置
                np_data = raw.get_data().T  # 转置为 (T, C)
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            return np_data, missing_mask

        try:
            np_data, missing_mask = _load_ts_data(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回一个小的占位符张量，避免程序崩溃
            return torch.zeros((1, 1), dtype=torch.float32), None

        # 最大长度240k
        max_length = 240000
        np_data = np_data[:max_length]
        # 用sklearn的SimpleImputer更简洁地填充NaN和inf
        if np_data.size > 0:
            # 先将inf/-inf替换为NaN
            np_data = np.where(np.isfinite(np_data), np_data, np.nan)
            # 如果全是NaN，直接返回占位符
            if np.isnan(np_data).all():
                return torch.zeros((1, 1), dtype=torch.float32), None
            # 用每列均值填充NaN
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            np_data = imputer.fit_transform(np_data)
        missing_mask = missing_mask[:max_length] if missing_mask is not None else None

        assert np_data.ndim == 2, f"Data must be 2D array with shape (T, C), got shape {np_data.shape}"
        
        # 验证数据类型
        if not np.issubdtype(np_data.dtype, np.number):
            raise ValueError(f"Data contains non-numeric values. Data type: {np_data.dtype}, sample data: {np_data[:5]}")
        
        # 如果启用了标准化，对数据进行标准化处理
        if self.use_scaler:
            # 为每个样本创建新的scaler实例，避免多进程冲突
            local_scaler = StandardScaler()
            if ori_file_path and os.path.exists(ori_file_path):
                # 如果提供了原始文件路径，使用原始数据进行fit_transform
                ori_np_data, ori_missing_mask = _load_ts_data(ori_file_path)
                if ori_np_data.ndim == 2:
                    local_scaler.fit(ori_np_data)
                    np_data = local_scaler.transform(np_data)
                else:
                    raise ValueError(f"Original data must be 2D array with shape (T, C), got shape {ori_np_data.shape}")
            else:
                np_data = local_scaler.fit_transform(np_data)
        
        # 统一转换为float32类型的tensor
        data = torch.from_numpy(np_data.astype(np.float32))
        mask = torch.from_numpy(missing_mask.astype(np.float32)) if missing_mask is not None else None

        # 根据unfold_channels参数决定是否展平维度
        if self.unfold_channels:
            # T, C = data.shape
            
            # 如果指定了patch_len和stride，需要确保patch不跨越不同的channel
            # if self.patch_len is not None:
            #     padding = torch.zeros(self.patch_len, C)
            #     data = torch.cat([data, padding], dim=0)
            #     T = data.shape[0]  # 更新时间维度

            # 按通道顺序展平：先第1个通道，然后第2个通道，以此类推
            # 将 (T, C) 转置为 (C, T)，然后展平为 (C*T, 1)
            # data = data.transpose(0, 1).reshape(-1, 1)  # 从 (T, C) -> (C, T) -> (C*T, 1)
            data = data.reshape(-1, 1)  # 直接展平为 (T*C, 1)
            mask = mask.reshape(-1, 1) if mask is not None else None   # mask也展平为 (T*C, 1)
        
        return data, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_text = random.choice(sample['input_text'])
        gt_text = random.choice(sample.get("gt_text", [""]))
        input_text_chat = {"role": "user", "content": input_text}

        if "gt_ts" in sample and sample["gt_ts"]:
            # 预测任务：只有输入，没有目标文本
            input_ids = self.tokenizer.apply_chat_template(
                [input_text_chat],
                tokenize=True,
                truncation=True,
                max_length=self.max_tokens
            )
            # input_ids = None
            labels = None  # 预测时不需要labels
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
            # 合并 input + target，用于 causal LM 训练
            input_ids = input_encoding + target_encoding
            labels = [-100] * len(input_encoding) + target_encoding  # 仅计算 Assistant 部分 loss
            labels[-1] = -100 # 最后一个 token 不计算 loss

        # input_ts_path = sample["input_ts"]["path"]
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
                    # ori_ts_path = input_ts['original']['ori_path']
                    # if self.base_dir:
                    #     ori_ts_path = os.path.join(self.base_dir, ori_ts_path)
                    # input_ts_tensor, input_ts_mask = self.load_ts_data(input_ts_path, ori_file_path=ori_ts_path)
                    input_ts_tensor, input_ts_mask = self.load_ts_data(input_ts_path)
                else:
                    input_ts_tensor, input_ts_mask = self.load_ts_data(input_ts_path)  # 返回data和mask
        else:
            input_ts_path = None
            # input_ts_tensor = torch.zeros((1, 1), dtype=torch.float32)  # 占位符，避免None
            # input_ts_tensor = torch.randn((1, 1), dtype=torch.float32)  # 占位符，避免None
            input_ts_tensor = torch.ones((1, 1), dtype=torch.float32) * 10 # 占位符，避免None
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

        # 如果有预测目标路径，也加载预测音频
        if "gt_ts" in sample and sample["gt_ts"]:
            gt_ts_path = sample["gt_ts"]["path"]
            if self.base_dir:
                gt_ts_path = os.path.join(self.base_dir, gt_ts_path)
            
            gt_ts = None
            gt_ts_mask = None
            if gt_ts_path:
                gt_ts, gt_ts_mask = self.load_ts_data(gt_ts_path)  # 返回tensor和mask
            
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
        对时间序列张量列表进行padding，统一通道数和时间维度
        
        :param ts_list: 时间序列张量列表，每个张量形状为 (T, C)
        :return: padding后的张量，形状为 (Batch, Max_Time, Max_Channels)
        """
        if not ts_list or any(ts is None for ts in ts_list):
            return None
            
        # 统一最大长度
        max_channels = max(ts.shape[1] for ts in ts_list if ts is not None)
        max_time = max(ts.shape[0] for ts in ts_list if ts is not None)

        padded_ts_list = []
        for ts in ts_list:
            # 先在通道维度上对齐到 max_channels
            T, C = ts.shape
            if C < max_channels:
                add_c = max_channels - C
                pad_c = torch.zeros(T, add_c, dtype=ts.dtype, device=ts.device)
                if self.pad_side_channel == 'left':
                    ts = torch.cat([pad_c, ts], dim=1)
                else:  # default: right
                    ts = torch.cat([ts, pad_c], dim=1)

            # 再在时间维度上对齐到 max_time
            T = ts.shape[0]
            if T < max_time:
                add_t = max_time - T
                pad_t = torch.zeros(add_t, ts.shape[1], dtype=ts.dtype, device=ts.device)
                if self.pad_side_time == 'left':
                    ts = torch.cat([pad_t, ts], dim=0)
                else:  # default: right
                    ts = torch.cat([ts, pad_t], dim=0)

            padded_ts_list.append(ts)

        # 堆叠为批量张量: (Batch, Max_Time, Max_Channels)
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

        # Dynamic padding - 处理input_ids为None的情况
        input_ids_tensor = None
        if None not in input_ids:
            input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
        
        # 处理labels，如果有None则不进行padding
        labels_tensor = None
        if None not in labels:
            labels_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(lbls, dtype=torch.long) for lbls in labels],
                batch_first=True,
                padding_value=-100
            )

        # 处理输入时间序列数据 (T, C) 格式
        # input_ts_tensors = self._pad_ts_tensors(input_ts_list)

        result = {
            "input_ids": input_ids_tensor,  # 可能为None
            "labels": labels_tensor,  # 可能为None
            # "input_ts": input_ts_tensors,
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

        # 如果有预测目标时间序列数据，也进行处理
        if "gt_ts" in batch[0]:
            gt_ts_list = [item["gt_ts"] for item in batch]
            # gt_ts_tensors = self._pad_ts_tensors(gt_ts_list)
            result['gt_ts_list'] = gt_ts_list
            # result["gt_ts"] = gt_ts_tensors
            result["gt_ts_paths"] = [item.get("gt_ts_path") for item in batch]
            result["gt_ts_mask_list"] = [item.get("gt_ts_mask", None) for item in batch]

        return result
