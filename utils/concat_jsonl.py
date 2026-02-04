#!/usr/bin/env python3
"""
JSONL文件合并脚本
将多个JSONL文件合并成一个文件
支持对大文件进行随机采样
"""

import os
import json
import random
from pathlib import Path

def concat_jsonl_files(input_files, output_file, subset=False, max_lines_per_file=20000, random_seed=42):
    """
    合并多个JSONL文件到一个输出文件
    
    Args:
        input_files (list): 输入JSONL文件路径列表
        output_file (str): 输出文件路径
        subset (bool): 是否对超过max_lines_per_file的文件进行随机采样
        max_lines_per_file (int): 每个文件的最大行数，超过时进行采样
        random_seed (int): 随机种子，确保可复现
    """
    # 设置随机种子确保可复现
    random.seed(random_seed)
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 - {file_path}")
                continue
                
            print(f"正在处理: {file_path}")
            file_lines = 0
            
            try:
                # 首先读取所有有效的行
                valid_lines = []
                with open(file_path, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # 跳过空行
                            # 验证是否为有效的JSON
                            try:
                                json.loads(line)
                                valid_lines.append(line)
                            except json.JSONDecodeError as e:
                                print(f"警告: 跳过无效JSON行 in {file_path}: {e}")
                
                # 如果启用subset且行数超过限制，进行随机采样
                if subset and len(valid_lines) > max_lines_per_file:
                    print(f"  文件有 {len(valid_lines)} 行，超过 {max_lines_per_file} 行限制，进行随机采样...")
                    sampled_lines = random.sample(valid_lines, max_lines_per_file)
                    lines_to_write = sampled_lines
                    print(f"  采样到 {len(lines_to_write)} 行")
                else:
                    lines_to_write = valid_lines
                
                # 写入选定的行
                for line in lines_to_write:
                    outf.write(line + '\n')
                    file_lines += 1
                    total_lines += 1
                
                print(f"  最终写入 {file_lines} 行")
                
            except Exception as e:
                print(f"错误: 处理文件 {file_path} 时出错: {e}")
    
    print(f"\n合并完成! 总共写入 {total_lines} 行到 {output_file}")
    if subset:
        print(f"大文件采样限制: {max_lines_per_file} 行")
        print(f"使用随机种子: {random_seed}")

def main():
    base_dir = "dataset/Release_train_standard"
    # 定义要合并的JSONL文件列表 - 之前的预测任务文件
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
    
    # 新的分类和异常检测任务文件列表
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
    
    # 输出文件路径
    output_file = "dataset/merged_train_forecast_v3.jsonl"

    # 设置subset参数
    subset = False  # 改为True启用采样功能
    max_lines_per_file = 20000  # 每个文件最大行数
    random_seed = 42  # 固定随机种子确保可复现
    
    print(f"开始合并 {len(input_files)} 个JSONL文件...")
    print(f"输出文件: {output_file}")
    print(f"Subset模式: {'启用' if subset else '禁用'}")
    if subset:
        print(f"采样限制: {max_lines_per_file} 行/文件")
        print(f"随机种子: {random_seed}")
    print("-" * 50)
    
    # 检查输入文件是否存在
    existing_files = []
    for file_path in input_files:
        file_path = os.path.join(base_dir, file_path)
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"文件不存在: {file_path}")
    
    if not existing_files:
        print("错误: 没有找到任何输入文件!")
        return
    
    print(f"找到 {len(existing_files)} 个有效文件")
    print("-" * 50)
    
    # 合并文件
    concat_jsonl_files(existing_files, output_file, subset=subset, max_lines_per_file=max_lines_per_file, random_seed=random_seed)
    
    # 显示输出文件信息
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"输出文件大小: {file_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
