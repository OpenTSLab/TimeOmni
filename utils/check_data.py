import os
import json
import numpy as np
import pandas as pd
import torchaudio
import mne
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import logging
from typing import Tuple, Optional, Dict, Any


def _load_ts_data(file_path: str) -> np.ndarray:
    """
    内部函数，根据文件后缀读取时间序列数据
    支持格式：
    - 音频文件：.wav, .mp3, .flac, .m4a (使用 torchaudio)
    - CSV文件：.csv (假设每列是一个通道)
    - NumPy文件：.npy
    - MNE文件：.fif (EEG/MEG数据)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.wav', '.mp3', '.flac', '.m4a']:
        # 音频文件：使用torchaudio
        waveform, _ = torchaudio.load(file_path)
        # torchaudio返回格式为 (C, T)，需要转置为 (T, C)
        np_data = waveform.transpose(0, 1).numpy()
        
    elif file_ext == '.csv':
        # CSV文件：假设每列是一个通道
        df = pd.read_csv(file_path, header=None)
        
        # 检测并处理'X'缺失值
        if (df == 'X').any().any() or (df == 'x').any().any():
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
        # 如果是2D但形状为 (C, T)，转置为 (T, C)
        # elif np_data.ndim == 2 and np_data.shape[0] < np_data.shape[1]:
        #     # 假设通道数小于时间步数，进行转置
        #     np_data = np_data.T
            
    elif file_ext == '.fif':
        # 读取raw数据
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        # 获取数据：MNE返回 (n_channels, n_times)，需要转置
        np_data = raw.get_data().T  # 转置为 (T, C)
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # 检查异常值
    if np.isnan(np_data).any():
        raise ValueError(f"Data contains NaN values in file: {file_path}")
    if np.isinf(np_data).any():
        raise ValueError(f"Data contains Inf values in file: {file_path}")

    return np_data


def process_single_line(line_data: Tuple[int, str], base_dir: str) -> Dict[str, Any]:
    """
    处理单行JSONL数据
    返回处理结果的字典
    """
    line_num, line = line_data
    result = {
        'line_num': line_num,
        'success': True,
        'errors': [],
        'input_ts_info': None,
        'gt_ts_info': None
    }
    
    try:
        data = json.loads(line.strip())
        
        # Analyze input_ts
        input_ts = data.get('input_ts', {})
        if input_ts:
            already_segment = input_ts.get('already_segment', None)
            
            if already_segment is False:
                original = input_ts.get('original', {})
                input_ts_path = original.get('ori_path', None) if original else None
            elif already_segment is True:
                seg = input_ts.get('segment', {})
                input_ts_path = seg.get('seg_path', None) if seg else None
            else:
                input_ts_path = None

            if input_ts_path:
                try:
                    if not os.path.isabs(input_ts_path) and base_dir:
                        input_ts_path = os.path.join(base_dir, input_ts_path)
                    input_data = _load_ts_data(input_ts_path)
                    result['input_ts_info'] = {
                        'path': input_ts_path,
                        'shape': input_data.shape,
                        'dtype': str(input_data.dtype),
                        'has_nan': bool(np.isnan(input_data).any()),
                        'has_inf': bool(np.isinf(input_data).any())
                    }
                except Exception as e:
                    error_msg = f"Failed to load input_ts from {input_ts_path}: {e}"
                    result['errors'].append(error_msg)

        # Analyze gt_ts
        gt_ts = data.get('gt_ts', None)
        if gt_ts:
            gt_ts_path = gt_ts.get('path', None)
            if gt_ts_path and not os.path.isabs(gt_ts_path) and base_dir:
                gt_ts_path = os.path.join(base_dir, gt_ts_path)
            if gt_ts_path:
                try:
                    gt_data = _load_ts_data(gt_ts_path)
                    result['gt_ts_info'] = {
                        'path': gt_ts_path,
                        'shape': gt_data.shape,
                        'dtype': str(gt_data.dtype),
                        'has_nan': bool(np.isnan(gt_data).any()),
                        'has_inf': bool(np.isinf(gt_data).any())
                    }
                except Exception as e:
                    error_msg = f"Failed to load gt_ts from {gt_ts_path}: {e}"
                    result['errors'].append(error_msg)
                    
        if result['errors']:
            result['success'] = False
            
    except json.JSONDecodeError as e:
        result['success'] = False
        result['errors'].append(f"JSON decode error: {e}")
    except Exception as e:
        result['success'] = False
        result['errors'].append(f"Unexpected error: {e}")
    
    return result


def check_data_with_multiprocessing(file_path: str, num_workers: int = None, output_file: str = None, base_dir: str = None) -> Dict[str, Any]:
    """
    使用多进程检查JSONL数据文件
    
    Args:
        file_path: JSONL文件路径
        num_workers: 进程数，默认为CPU核数
        output_file: 输出结果文件路径，可选
    
    Returns:
        包含统计信息的字典
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始检查数据文件: {file_path}")
    logger.info(f"使用进程数: {num_workers}")
    
    # 首先读取所有行，获取总行数
    logger.info("读取文件并计算总行数...")
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                lines.append((line_num, line))
        
        total_lines = len(lines)
        logger.info(f"总行数: {total_lines}")
        
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return {'error': f'File not found: {file_path}'}
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return {'error': f'Failed to read file: {e}'}
    
    # 初始化结果统计
    stats = {
        'total_lines': total_lines,
        'processed_lines': 0,
        'successful_lines': 0,
        'failed_lines': 0,
        'input_ts_stats': {
            'total_count': 0,
            'valid_count': 0,
            'shapes': {},
            'dtypes': {},
            'nan_count': 0,
            'inf_count': 0
        },
        'gt_ts_stats': {
            'total_count': 0,
            'valid_count': 0,
            'shapes': {},
            'dtypes': {},
            'nan_count': 0,
            'inf_count': 0
        },
        'errors': []
    }
    
    # 使用进程池处理数据
    logger.info("开始多进程处理...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_line = {executor.submit(process_single_line, line_data, base_dir): line_data[0] 
                         for line_data in lines}
        
        # 使用tqdm显示进度
        with tqdm(total=total_lines, desc="处理进度", unit="行") as pbar:
            for future in as_completed(future_to_line):
                try:
                    result = future.result()
                    stats['processed_lines'] += 1
                    
                    if result['success']:
                        stats['successful_lines'] += 1
                    else:
                        stats['failed_lines'] += 1
                        for error in result['errors']:
                            stats['errors'].append(f"Line {result['line_num']}: {error}")
                    
                    # 更新输入时间序列统计
                    if result['input_ts_info']:
                        stats['input_ts_stats']['total_count'] += 1
                        stats['input_ts_stats']['valid_count'] += 1
                        
                        shape_str = str(result['input_ts_info']['shape'])
                        stats['input_ts_stats']['shapes'][shape_str] = stats['input_ts_stats']['shapes'].get(shape_str, 0) + 1
                        
                        dtype = result['input_ts_info']['dtype']
                        stats['input_ts_stats']['dtypes'][dtype] = stats['input_ts_stats']['dtypes'].get(dtype, 0) + 1
                        
                        if result['input_ts_info']['has_nan']:
                            stats['input_ts_stats']['nan_count'] += 1
                        if result['input_ts_info']['has_inf']:
                            stats['input_ts_stats']['inf_count'] += 1
                    
                    # 更新真值时间序列统计
                    if result['gt_ts_info']:
                        stats['gt_ts_stats']['total_count'] += 1
                        stats['gt_ts_stats']['valid_count'] += 1
                        
                        shape_str = str(result['gt_ts_info']['shape'])
                        stats['gt_ts_stats']['shapes'][shape_str] = stats['gt_ts_stats']['shapes'].get(shape_str, 0) + 1
                        
                        dtype = result['gt_ts_info']['dtype']
                        stats['gt_ts_stats']['dtypes'][dtype] = stats['gt_ts_stats']['dtypes'].get(dtype, 0) + 1
                        
                        if result['gt_ts_info']['has_nan']:
                            stats['gt_ts_stats']['nan_count'] += 1
                        if result['gt_ts_info']['has_inf']:
                            stats['gt_ts_stats']['inf_count'] += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    line_num = future_to_line[future]
                    error_msg = f"Line {line_num}: Processing failed: {e}"
                    stats['errors'].append(error_msg)
                    stats['failed_lines'] += 1
                    stats['processed_lines'] += 1
                    pbar.update(1)
                    logger.error(error_msg)
    
    # 输出统计结果
    logger.info("数据检查完成！")
    logger.info(f"总行数: {stats['total_lines']}")
    logger.info(f"成功处理: {stats['successful_lines']}")
    logger.info(f"失败行数: {stats['failed_lines']}")
    logger.info(f"输入时间序列有效数量: {stats['input_ts_stats']['valid_count']}")
    logger.info(f"真值时间序列有效数量: {stats['gt_ts_stats']['valid_count']}")
    
    if stats['errors']:
        logger.warning(f"发现 {len(stats['errors'])} 个错误")
        if len(stats['errors']) <= 10:
            for error in stats['errors']:
                logger.warning(error)
        else:
            logger.warning("错误过多，仅显示前10个:")
            for error in stats['errors'][:10]:
                logger.warning(error)
    
    # 保存详细结果到文件
    if output_file:
        logger.info(f"保存详细结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='多进程检查时间序列数据文件')
    parser.add_argument('--file', '-f', type=str, 
                       default='dataset/merged_train_forecast.jsonl',
                       help='JSONL文件路径')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='进程数，默认为CPU核数')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出结果文件路径')
    parser.add_argument('--base_dir', '-b', type=str, default='dataset/Release_v1',
                       help='基础目录，若提供则将相对路径转换为绝对路径')
    
    args = parser.parse_args()
    
    # 检查数据
    stats = check_data_with_multiprocessing(
        file_path=args.file,
        num_workers=args.workers,
        output_file=args.output,
        base_dir=args.base_dir
    )
    
    # 输出简要统计
    if 'error' not in stats:
        print("\n" + "="*60)
        print("数据检查统计摘要")
        print("="*60)
        print(f"总行数: {stats['total_lines']}")
        print(f"成功处理: {stats['successful_lines']}")
        print(f"失败行数: {stats['failed_lines']}")
        print(f"成功率: {stats['successful_lines']/stats['total_lines']*100:.2f}%")
        print("\n输入时间序列统计:")
        print(f"  有效数量: {stats['input_ts_stats']['valid_count']}")
        print(f"  包含NaN: {stats['input_ts_stats']['nan_count']}")
        print(f"  包含Inf: {stats['input_ts_stats']['inf_count']}")
        print("  形状分布:", dict(list(stats['input_ts_stats']['shapes'].items())[:5]))
        print("\n真值时间序列统计:")
        print(f"  有效数量: {stats['gt_ts_stats']['valid_count']}")
        print(f"  包含NaN: {stats['gt_ts_stats']['nan_count']}")
        print(f"  包含Inf: {stats['gt_ts_stats']['inf_count']}")
        print("  形状分布:", dict(list(stats['gt_ts_stats']['shapes'].items())[:5]))
        print("="*60)


if __name__ == "__main__":
    main()