#!/usr/bin/env python3
"""
Script to analyze JSONL files in Release_v1 directory structure.
Collects statistics on:
- input_ts ori_length (when already_segment is false)
- input_text word count
- gt_text word count  
- gt_ts length
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def count_words(text_list):
    """Count words in a list of text strings, handling None values."""
    if not text_list:
        return 0
    
    total_words = 0
    for text in text_list:
        if text is not None and isinstance(text, str):
            # Simple word count by splitting on whitespace
            total_words += len(text.split())
    total_words /= len(text_list)  # Average words if multiple texts
    return total_words


def analyze_jsonl_file(file_path):
    """Analyze a single JSONL file and return statistics."""
    stats = {
        'file_path': str(file_path),
        'total_records': 0,
        'channel': [],
        'input_ts_length': [],
        'input_text_word_counts': [],
        'gt_text_word_counts': [],
        'gt_ts_lengths': [],
        'already_segment_true_count': 0,
        'already_segment_false_count': 0,
        'gt_ts_none_count': 0,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    stats['total_records'] += 1
                    
                    # Analyze input_ts
                    input_ts = data.get('input_ts', {})
                    already_segment = input_ts.get('already_segment', None) if input_ts else None
                    channel = input_ts.get('channel', 1) if input_ts else 0
                    stats['channel'].append(channel)
                    
                    if already_segment is False:
                        stats['already_segment_false_count'] += 1
                        original = input_ts.get('original', {})
                        input_ts_length = original.get('ori_length', None)
                        input_ts_length *= channel  # Adjust length by channel count
                        if input_ts_length is not None:
                            stats['input_ts_length'].append(input_ts_length)
                    elif already_segment is True:
                        stats['already_segment_true_count'] += 1
                        seg = input_ts.get('segment', {})
                        input_ts_length = seg.get('seg_length', None)
                        input_ts_length *= channel  # Adjust length by channel count
                        if input_ts_length is not None:
                            stats['input_ts_length'].append(input_ts_length)

                    # Analyze input_text
                    input_text = data.get('input_text', [])
                    input_word_count = count_words(input_text)
                    stats['input_text_word_counts'].append(input_word_count)
                    
                    # Analyze gt_text
                    gt_text = data.get('gt_text', [])
                    gt_word_count = count_words(gt_text)
                    stats['gt_text_word_counts'].append(gt_word_count)
                    
                    # Analyze gt_ts
                    gt_ts = data.get('gt_ts', None)
                    if gt_ts is None:
                        stats['gt_ts_none_count'] += 1
                    else:
                        gt_ts_length = gt_ts.get('length', None)
                        gt_ts
                        if gt_ts_length is not None:
                            stats['gt_ts_lengths'].append(gt_ts_length)
                            
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {file_path} at line {line_num}: {e}")
                    continue
                except Exception as e:
                    # print(f"Error processing line {line_num} in {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                    
    except Exception as e:
        # print(f"Error reading file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return stats


def calculate_summary_stats(values):
    """Calculate summary statistics for a list of values."""
    if not values:
        return {
            # 'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None
        }
    
    import statistics
    return {
        # 'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': round(statistics.mean(values), 2),
        'median': statistics.median(values)
    }


def find_jsonl_files(root_dir):
    """Find all JSONL files in subdirectories of root_dir."""
    jsonl_files = []
    root_path = Path(root_dir)
    # 如果路径包含 'train'，只遍历该目录下的 jsonl 文件
    if 'train' in str(root_path).lower():
        for file_path in root_path.glob("*.jsonl"):
            jsonl_files.append(file_path)
    else:
        for subdir in root_path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.glob("*.jsonl"):
                    jsonl_files.append(file_path)
    return sorted(jsonl_files)


def main():
    parser = argparse.ArgumentParser(description='Analyze JSONL files in Release_v1 directory')
    parser.add_argument('--root_dir', default='dataset/Release_v1',
                       help='Root directory containing subdirectories with JSONL files')
    parser.add_argument('--output_csv', default='./dataset/release_v1_analysis.csv',
                       help='Output CSV file name')
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist")
        return
    
    print(f"Analyzing JSONL files in: {root_dir}")
    
    # Find all JSONL files
    jsonl_files = find_jsonl_files(root_dir)
    print(f"Found {len(jsonl_files)} JSONL files")
    
    if not jsonl_files:
        print("No JSONL files found!")
        return
    
    # Analyze each file
    all_stats = []
    overall_stats = defaultdict(list)
    
    for file_path in jsonl_files:
        print(f"Processing: {file_path}")
        stats = analyze_jsonl_file(file_path)
        if stats:
            all_stats.append(stats)
            
            # Aggregate overall statistics
            overall_stats['input_ts_length'].extend(stats['input_ts_length'])
            overall_stats['input_text_word_counts'].extend(stats['input_text_word_counts'])
            overall_stats['gt_text_word_counts'].extend(stats['gt_text_word_counts'])
            overall_stats['gt_ts_lengths'].extend(stats['gt_ts_lengths'])
    
    # Create detailed results DataFrame
    detailed_results = []
    for stats in all_stats:
        result = {
            'file_path': stats['file_path'],
            'dataset_name': Path(stats['file_path']).parent.name,
            'total_records': stats['total_records'],
            'already_segment_false_count': stats['already_segment_false_count'],
            'already_segment_true_count': stats['already_segment_true_count'],
            'gt_ts_none_count': stats['gt_ts_none_count'],
        }
        
        # Add summary statistics
        channel_stats = calculate_summary_stats(stats['channel'])
        result.update({f'channel_{k}': v for k, v in channel_stats.items()})

        length_stats = calculate_summary_stats(stats['input_ts_length'])
        result.update({f'input_ts_length_{k}': v for k, v in length_stats.items()})

        input_text_stats = calculate_summary_stats(stats['input_text_word_counts'])
        result.update({f'input_text_words_{k}': v for k, v in input_text_stats.items()})
        
        gt_text_stats = calculate_summary_stats(stats['gt_text_word_counts'])
        result.update({f'gt_text_words_{k}': v for k, v in gt_text_stats.items()})
        
        gt_ts_stats = calculate_summary_stats(stats['gt_ts_lengths'])
        result.update({f'gt_ts_length_{k}': v for k, v in gt_ts_stats.items()})
        
        detailed_results.append(result)
    
    # Save detailed results to CSV
    df = pd.DataFrame(detailed_results)
    df.to_csv(args.output_csv, index=False)
    print(f"Detailed results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
