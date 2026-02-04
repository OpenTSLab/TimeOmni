#!/usr/bin/env python3
"""
JSONL merge script.
Merge multiple JSONL files into a single file.
Supports random sampling for large files.
"""

import os
import json
import random
from pathlib import Path

def concat_jsonl_files(input_files, output_file, subset=False, max_lines_per_file=20000, random_seed=42):
    """
    Merge multiple JSONL files into one output file.

    Args:
        input_files (list): list of input JSONL file paths
        output_file (str): output file path
        subset (bool): whether to randomly sample files exceeding max_lines_per_file
        max_lines_per_file (int): max lines per file before sampling
        random_seed (int): random seed for reproducibility
    """
    # Set random seed for reproducibility.
    random.seed(random_seed)
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"Warning: file not found - {file_path}")
                continue
                
            print(f"Processing: {file_path}")
            file_lines = 0
            
            try:
                # First, read all valid lines.
                valid_lines = []
                with open(file_path, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # Skip empty lines.
                            # Validate JSON line.
                            try:
                                json.loads(line)
                                valid_lines.append(line)
                            except json.JSONDecodeError as e:
                                print(f"Warning: skipping invalid JSON line in {file_path}: {e}")
                
                # If subset is enabled and line count exceeds limit, sample randomly.
                if subset and len(valid_lines) > max_lines_per_file:
                    print(f"  File has {len(valid_lines)} lines, exceeds {max_lines_per_file}; sampling...")
                    sampled_lines = random.sample(valid_lines, max_lines_per_file)
                    lines_to_write = sampled_lines
                    print(f"  Sampled {len(lines_to_write)} lines")
                else:
                    lines_to_write = valid_lines
                
                # Write selected lines.
                for line in lines_to_write:
                    outf.write(line + '\n')
                    file_lines += 1
                    total_lines += 1
                
                print(f"  Wrote {file_lines} lines")
                
            except Exception as e:
                print(f"Error: failed to process file {file_path}: {e}")
    
    print(f"\nMerge completed! Wrote {total_lines} lines to {output_file}")
    if subset:
        print(f"Large-file sampling limit: {max_lines_per_file} lines")
        print(f"Random seed: {random_seed}")

def main():
    base_dir = "dataset/Release_train_standard"
    # Define JSONL files to merge - previous forecasting task files.
    input_files = [
        "Chaotic-Math-Forecasting-Chaotic_system-train-7656.jsonl",
        "ETT-Energy-Forecasting-ETT-train-21844.jsonl",
        # "FinMultiTime-Economics-Forecasting-Stock_closing_price-train-5035.jsonl",
        "FinMultiTime-Economics-Forecasting-Stock_closing_price-train-2712.jsonl",
        # "MT_bench-Economics-Forecasting-Stock_price-train-892.jsonl",
        "MT_bench-Economics-Forecasting-Stock_price-train-662.jsonl",
        # "MT_bench-Meteorology-Forecasting-Temperature-train-2742.jsonl",
        "MT_bench-Meteorology-Forecasting-Temperature-train-2464.jsonl",
        "MetroTraffic-Urbanism-Forecasting-Traffic-train-21816.jsonl",
        # "NewsForecast-Economics-Forecasting-Bitcoin_price-train-256.jsonl",
        "NewsForecast-Energy-Forecasting-Electronic_load-train-3655.jsonl",
        # "NewsForecast-Urbanism-Forecasting-Traffic_flow-train-172.jsonl",
        "NewsForecast-Urbanism-Forecasting-Traffic_flow-train-34.jsonl",
        # "TS_MQA-Meteorology-Forecasting-Weather-train-15.jsonl",
        # "TS_MQA-Meteorology-Imputation-Weather-train-19.jsonl",
        "TS_MQA-Neuroscience-Forecasting-EEG-train-3455.jsonl",
        "TS_MQA-Neuroscience-Imputation-EEG-train-3442.jsonl",
        "TS_MQA-Physiology-Forecasting-Health-train-13476.jsonl",
        "TS_MQA-Physiology-Imputation-Health-train-13474.jsonl",
        "textETT-Energy-Reverse_Forecasting-textETT-train-258010.jsonl"
    ]

    # input_files = [
    #     "FinMultiTime-Economics-Forecasting-Stock_closing_price-train-5035.jsonl",
    #     "MT_bench-Economics-Forecasting-Stock_price-train-892.jsonl",
    #     "MT_bench-Meteorology-Forecasting-Temperature-train-2742.jsonl",
    #     "NewsForecast-Economics-Forecasting-Bitcoin_price-train-256.jsonl",
    #     "NewsForecast-Urbanism-Forecasting-Traffic_flow-train-172.jsonl",
    # ]
    
    # New classification and anomaly detection task files list.
    # input_files = [
    #     "CWRU-Manufacturing-Classification-Industrial_bearings-train-3600.jsonl",
    #     "GWOSC_GW_Event-Astronomy-Event_Detection-Gravitational_Wave-train-79716.jsonl",
    #     "LEAVES-Astronomy-Classification-Light_Curve-train-201259.jsonl",
    #     "MDD-Neuroscience-Anomaly_Detection-Depressive_disorder-train-6462.jsonl",
    #     # "MIMII_Due-Manufacturing-Anomaly_Detection-Industrial_machine-train-30212",
    #     "MT_bench-Economics-QA-Stock-train-704.jsonl",
    #     "MT_bench-Meteorology-QA-Temperature-train-960.jsonl",
    #     "MarmAudio-Bioacoustics-Classification-Marmoset-train-214100.jsonl",
    #     "PTB_XL-Physiology-Classification-ECG-train-19599.jsonl",
    #     "Powdermill-Bioacoustics-Classification-Birds_sound-train-14450.jsonl",
    #     "RadSeg-Radar-Classification-Radar_segment-train-84163.jsonl",
    #     "RadarCom-Radar-Classification-Radar_signal-train-12537.jsonl",
    #     "STEAD-Earth_Science-Event_Detection-Earthquake-train-201259.jsonl",
    #     "Sleep-Neuroscience-Classification-Sleep_stage-train-157946.jsonl",
    #     # "TIMECAP-Meteorology-Anomaly_Detection-Rainfall-train-4521.jsonl",
    #     "TIMECAP-Meteorology-Anomaly_Detection-Rainfall-train-6638-upsample.jsonl",
    #     "TS_MQA-Meteorology-Anomaly_Detection-Weather-train-11290.jsonl",
    #     "TS_MQA-Physiology-Anomaly_Detection-ECG-train-9051.jsonl",
    #     # "TS_MQA-Physiology-Anomaly_Detection-Freezing-train-16618.jsonl",
    #     "TS_MQA-Physiology-Anomaly_Detection-Freezing-train-29878-upsample.jsonl",
    #     "TS_MQA-Physiology-Classification-Activity-train-16682.jsonl",
    #     "TUEV-Neuroscience-Classification-EEG_waveform-train-1592.jsonl",
    #     "WBCIC_SHU-Neuroscience-Classification-Movement_imagination-train-36890.jsonl",
    #     "iNaturalist-Bioacoustics-Classification-Animal_sound-train-171654.jsonl"
    # ]
    
    # Output file path.
    output_file = "dataset/merged_train_forecast_v3.jsonl"

    # Subset parameters.
    subset = False  # Set to True to enable sampling.
    max_lines_per_file = 20000  # Max lines per file.
    random_seed = 42  # Fixed random seed for reproducibility.
    
    print(f"Starting merge of {len(input_files)} JSONL files...")
    print(f"Output file: {output_file}")
    print(f"Subset mode: {'enabled' if subset else 'disabled'}")
    if subset:
        print(f"Sampling limit: {max_lines_per_file} lines/file")
        print(f"Random seed: {random_seed}")
    print("-" * 50)
    
    # Check input file existence.
    existing_files = []
    for file_path in input_files:
        file_path = os.path.join(base_dir, file_path)
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"File not found: {file_path}")
    
    if not existing_files:
        print("Error: no input files found!")
        return
    
    print(f"Found {len(existing_files)} valid files")
    print("-" * 50)
    
    # Merge files.
    concat_jsonl_files(existing_files, output_file, subset=subset, max_lines_per_file=max_lines_per_file, random_seed=random_seed)
    
    # Show output file info.
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Output file size: {file_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
