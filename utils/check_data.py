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
    Internal helper to load time-series data by file extension.
    Supported formats:
    - Audio: .wav, .mp3, .flac, .m4a (via torchaudio)
    - CSV: .csv (each column is a channel)
    - NumPy: .npy
    - MNE: .fif (EEG/MEG data)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.wav', '.mp3', '.flac', '.m4a']:
        # Audio file: use torchaudio.
        waveform, _ = torchaudio.load(file_path)
        # torchaudio returns (C, T), transpose to (T, C).
        np_data = waveform.transpose(0, 1).numpy()
        
    elif file_ext == '.csv':
        # CSV file: assume each column is a channel.
        df = pd.read_csv(file_path, header=None)
        
        # Detect and handle 'X' missing values.
        if (df == 'X').any().any() or (df == 'x').any().any():
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
        # If 2D and shaped (C, T), transpose to (T, C).
        # elif np_data.ndim == 2 and np_data.shape[0] < np_data.shape[1]:
        #     # Assume channels < time steps, then transpose.
        #     np_data = np_data.T
            
    elif file_ext == '.fif':
        # Read raw data.
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        # MNE returns (n_channels, n_times), transpose it.
        np_data = raw.get_data().T  # Transpose to (T, C).
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Check for invalid values.
    if np.isnan(np_data).any():
        raise ValueError(f"Data contains NaN values in file: {file_path}")
    if np.isinf(np_data).any():
        raise ValueError(f"Data contains Inf values in file: {file_path}")

    return np_data


def process_single_line(line_data: Tuple[int, str], base_dir: str) -> Dict[str, Any]:
    """
    Process a single JSONL line.
    Returns a dict with processing results.
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
    Check a JSONL data file using multiprocessing.

    Args:
        file_path: JSONL file path
        num_workers: number of workers (defaults to CPU count)
        output_file: optional output file path
    
    Returns:
        Dict with statistics
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting data check: {file_path}")
    logger.info(f"Workers: {num_workers}")
    
    # Read all lines to get total count.
    logger.info("Reading file and counting total lines...")
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                lines.append((line_num, line))
        
        total_lines = len(lines)
        logger.info(f"Total lines: {total_lines}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {'error': f'File not found: {file_path}'}
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return {'error': f'Failed to read file: {e}'}
    
    # Initialize stats.
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
    
    # Process data with a process pool.
    logger.info("Starting multiprocessing...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks.
        future_to_line = {executor.submit(process_single_line, line_data, base_dir): line_data[0] 
                         for line_data in lines}
        
        # Use tqdm for progress display.
        with tqdm(total=total_lines, desc="Progress", unit="lines") as pbar:
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
                    
                    # Update input time-series stats.
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
                    
                    # Update ground-truth time-series stats.
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
    
    # Log summary.
    logger.info("Data check completed!")
    logger.info(f"Total lines: {stats['total_lines']}")
    logger.info(f"Successfully processed: {stats['successful_lines']}")
    logger.info(f"Failed lines: {stats['failed_lines']}")
    logger.info(f"Valid input time series: {stats['input_ts_stats']['valid_count']}")
    logger.info(f"Valid ground-truth time series: {stats['gt_ts_stats']['valid_count']}")
    
    if stats['errors']:
        logger.warning(f"Found {len(stats['errors'])} errors")
        if len(stats['errors']) <= 10:
            for error in stats['errors']:
                logger.warning(error)
        else:
            logger.warning("Too many errors; showing first 10:")
            for error in stats['errors'][:10]:
                logger.warning(error)
    
    # Save detailed results to file.
    if output_file:
        logger.info(f"Saving detailed results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Check time-series JSONL data with multiprocessing')
    parser.add_argument('--file', '-f', type=str, 
                       default='dataset/merged_train_forecast.jsonl',
                       help='JSONL file path')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='number of workers (defaults to CPU count)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='output file path for results')
    parser.add_argument('--base_dir', '-b', type=str, default='dataset/Release_v1',
                       help='base directory for resolving relative paths')
    
    args = parser.parse_args()
    
    # Run checks.
    stats = check_data_with_multiprocessing(
        file_path=args.file,
        num_workers=args.workers,
        output_file=args.output,
        base_dir=args.base_dir
    )
    
    # Print summary.
    if 'error' not in stats:
        print("\n" + "="*60)
        print("Data check summary")
        print("="*60)
        print(f"Total lines: {stats['total_lines']}")
        print(f"Successfully processed: {stats['successful_lines']}")
        print(f"Failed lines: {stats['failed_lines']}")
        print(f"Success rate: {stats['successful_lines']/stats['total_lines']*100:.2f}%")
        print("\nInput time-series stats:")
        print(f"  Valid count: {stats['input_ts_stats']['valid_count']}")
        print(f"  Contains NaN: {stats['input_ts_stats']['nan_count']}")
        print(f"  Contains Inf: {stats['input_ts_stats']['inf_count']}")
        print("  Shape distribution:", dict(list(stats['input_ts_stats']['shapes'].items())[:5]))
        print("\nGround-truth time-series stats:")
        print(f"  Valid count: {stats['gt_ts_stats']['valid_count']}")
        print(f"  Contains NaN: {stats['gt_ts_stats']['nan_count']}")
        print(f"  Contains Inf: {stats['gt_ts_stats']['inf_count']}")
        print("  Shape distribution:", dict(list(stats['gt_ts_stats']['shapes'].items())[:5]))
        print("="*60)


if __name__ == "__main__":
    main()