import json
import re
import numpy as np
import os
import argparse
import csv
from typing import List, Dict, Any


DETECTION_DICT = {
    "GWOSC GW Event": {"false": ['no', 'not'], "true": []},
    "MDD": {"false": ["healthy"], "true": ["depressive"]},
    "MIMII Due": {"false": ['normal'], "true": ['anomaly']},
    "STEAD": {"false": ['no', 'not'], "true": []},
    "TIMECAP": {"false": ['not'], "true": []},
    "TS_MQA": {"false": ['normal'], "true": ['anomaly']},
}


def classification_eval(data_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    评估分类任务和QA任务的准确率、UAR和F1分数，支持单选和多选分类
    
    Args:
        data_list: 从JSONL文件读取的数据列表，每个元素包含:
                  - generated_text: 模型生成的文本
                  - gt_result: 包含gt_class或answer的ground truth信息
                  
    Returns:
        Dict包含每个key的准确率、UAR和F1分数指标
    """
    if not data_list:
        return {}
    
    # 检查数据格式，确定是分类任务还是QA任务
    first_item = data_list[0]
    gt_result = first_item['gt_result']
    
    # 统一处理：将QA任务转换为分类任务的格式
    if 'answer' in gt_result:
        # QA任务格式: {"answer": "B"} -> 转换为 {"gt_class": {"answer": "B"}}
        all_keys = ['answer']
        is_qa_task = True
    elif 'gt_class' in gt_result:
        # 分类任务格式: {"gt_class": {"default": ["MI", "CD"]}}
        all_keys = gt_result['gt_class'].keys()
        is_qa_task = False
    else:
        return {}

    # 初始化计数器
    correct_counts = {key: 0 for key in all_keys}
    total_counts = {key: 0 for key in all_keys}

    # 收集所有可能的类别标签
    all_labels = {}
    for key in all_keys:
        all_labels[key] = set()
    
    # 先遍历一遍收集所有类别
    for item in data_list:
        gt_result = item['gt_result']
        
        # 统一处理QA和分类任务的数据格式
        if is_qa_task and 'answer' in gt_result:
            # QA任务：将answer转换为gt_class格式
            unified_gt_class = {'answer': gt_result['answer']}
        elif 'gt_class' in gt_result:
            # 分类任务：直接使用gt_class
            unified_gt_class = gt_result['gt_class']
        else:
            continue
            
        for key, gt_value in unified_gt_class.items():
            if isinstance(gt_value, list):
                all_labels[key].update([label.lower() for label in gt_value])
            else:
                all_labels[key].add(gt_value.lower())
    
    # 初始化每个类别的TP、FN和FP计数器
    class_tp = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    class_fn = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    class_fp = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    
    # 遍历每个样本
    for item in data_list:
        generated_text = item.get('generated_text', '').lower()  # 转小写便于比较  
        gt_result = item['gt_result']
        
        # 统一处理QA和分类任务的数据格式
        if is_qa_task and 'answer' in gt_result:
            # QA任务：将answer转换为gt_class格式
            unified_gt_class = {'answer': gt_result['answer']}
        elif 'gt_class' in gt_result:
            # 分类任务：直接使用gt_class
            unified_gt_class = gt_result['gt_class']
        else:
            continue
        
        # 统一处理逻辑
        for key, gt_value in unified_gt_class.items():
            total_counts[key] += 1
            
            # 处理单选和多选情况
            if isinstance(gt_value, list):
                # 多选情况：计算部分分数、UAR和F1
                correct_labels = 0
                total_labels = len(gt_value)
                
                # 收集预测为正的标签
                predicted_labels = set()
                for label in all_labels[key]:
                    if label in generated_text:
                        predicted_labels.add(label)
                
                gt_labels_set = set([label.lower() for label in gt_value])
                
                for label in gt_labels_set:
                    if label in predicted_labels:
                        correct_labels += 1
                        class_tp[key][label] += 1
                    else:
                        class_fn[key][label] += 1
                
                # 计算误报：预测为正但实际为负的标签
                for label in predicted_labels:
                    if label not in gt_labels_set:
                        class_fp[key][label] += 1
                
                # 按比例给分：正确预测的标签数 / 总标签数
                if total_labels > 0:
                    partial_score = correct_labels / total_labels
                    correct_counts[key] += partial_score
            else:
                # 单选情况：检查ground truth值是否在generated_text中
                gt_label_lower = gt_value.lower()
                predicted_labels = set()
                for label in all_labels[key]:
                    if label in generated_text:
                        predicted_labels.add(label)
                
                if gt_label_lower in predicted_labels:
                    correct_counts[key] += 1
                    class_tp[key][gt_label_lower] += 1
                else:
                    class_fn[key][gt_label_lower] += 1
                
                # 计算误报
                for label in predicted_labels:
                    if label != gt_label_lower:
                        class_fp[key][label] += 1
    
    # 计算准确率、UAR和F1分数
    results = {}
    for key in all_keys:
        # 确定结果键名的后缀
        if key in ['default', 'answer']:
            suffix = ''
        else:
            suffix = f'_{key}'
        
        if total_counts[key] > 0:
            results[f'accuracy{suffix}'] = correct_counts[key] / total_counts[key]
        else:
            results[f'accuracy{suffix}'] = 0.0
        
        # 统一的UAR和F1计算
        recalls = []
        precisions = []
        f1_scores = []
        
        for label in all_labels[key]:
            tp = class_tp[key][label]
            fn = class_fn[key][label]
            fp = class_fp[key][label]
            
            # 计算召回率
            if tp + fn > 0:
                recall = tp / (tp + fn)
                recalls.append(recall)
            
            # 计算精确率
            if tp + fp > 0:
                precision = tp / (tp + fp)
                precisions.append(precision)
            
            # 计算F1分数
            if tp + fn > 0 and tp + fp > 0:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)
        
        if recalls:
            results[f'uar{suffix}'] = np.mean(recalls)
        else:
            results[f'uar{suffix}'] = 0.0
        
        if f1_scores:
            results[f'f1_score{suffix}'] = np.mean(f1_scores)
        else:
            results[f'f1_score{suffix}'] = 0.0

    return results


def detect(generated_text, dataset_name):
    """
    根据DETECTION_DICT检测生成文本中的事件检测结果
    
    逻辑：
    1. 如果在生成内容中检测到false的list中的任意一个，就认为是false
    2. 否则就是true（如果true是空list）
    3. 如果true不是空的，就检测一下
    4. 如果两个都检测不到，需要返回None
    """
    if dataset_name not in DETECTION_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DETECTION_DICT")
    
    detection_config = DETECTION_DICT[dataset_name]
    false_indicators = detection_config.get("false", [])
    true_indicators = detection_config.get("true", [])
    
    generated_lower = generated_text.lower()
    
    # 1. 如果在生成内容中检测到false的list中的任意一个，就认为是false
    for indicator in false_indicators:
        if indicator.lower() in generated_lower:
            return False
    
    # 2. 如果true是空list，且没有检测到false指示词，则返回True
    if not true_indicators:
        return True
    
    # 3. 如果true不是空的，就检测一下
    for indicator in true_indicators:
        if indicator.lower() in generated_lower:
            return True
    
    # 4. 如果两个都检测不到，需要返回None
    return None


def extract_predicted_times(generated_text: str) -> List[int]:
    """
    从生成文本中提取所有预测的时间数值
    
    Args:
        generated_text: 模型生成的文本
        
    Returns:
        List[int]: 按出现顺序提取的时间数值列表
    """
    # 使用正则表达式提取数字
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        return [int(num) for num in numbers]
    return []


def detection_eval(data_list: List[Dict[str, Any]], dataset_name: str, task: str) -> Dict[str, float]:
    """
    评估检测任务的准确率、F1分数和相对误差
    
    Args:
        data_list: 从JSONL文件读取的数据列表，每个元素包含:
                  - generated_text: 模型生成的文本
                  - gt_result: 包含contain和多个时间字段的ground truth信息
        dataset_name: 数据集名称，用于获取对应的检测字典
        task: 任务类型，"Event Detection"或"Anomaly Detection"
                  
    Returns:
        Dict包含检测准确率、F1分数和相对误差中位数
    """
    if not data_list:
        return {}
    
    correct_detection = 0
    total_samples = 0
    relative_errors = []
    
    # F1分数计算需要的统计量
    true_positive = 0   # 真正例：实际为正，预测为正
    false_positive = 0  # 假正例：实际为负，预测为正
    true_negative = 0   # 真负例：实际为负，预测为负
    false_negative = 0  # 假负例：实际为正，预测为负
    
    # 成功率：能做出明确判断（非None）的比例
    success_count = 0
    
    for item in data_list:
        generated_text = item.get('generated_text', '')
        
        if 'gt_result' not in item:
            continue
            
        gt_result = item['gt_result']
        gt_contain = gt_result.get('contain', False)
        
        total_samples += 1
        
        predicted_contain = detect(item['generated_text'], dataset_name)
        
        # 如果无法判断，按失败计入（保持在分母中且不增加correct_detection）
        if predicted_contain is None:
            # 不参与F1的混淆矩阵，不计算相对误差
            continue

        # 能判断则计为一次成功
        success_count += 1
        
        detection_correct = (predicted_contain == gt_contain)
        if detection_correct:
            correct_detection += 1
        
        # 统计F1分数需要的混淆矩阵元素
        if gt_contain == True and predicted_contain == True:
            true_positive += 1
        elif gt_contain == False and predicted_contain == True:
            false_positive += 1
        elif gt_contain == False and predicted_contain == False:
            true_negative += 1
        elif gt_contain == True and predicted_contain == False:
            false_negative += 1
            
        # 只有在Event Detection任务且检测正确且包含事件时才进行数字抓取
        if task == "Event Detection" and gt_contain and predicted_contain:
            predicted_times = extract_predicted_times(generated_text)
            
            gt_time_fields = []
            for key, value in gt_result.items():
                if key != 'contain' and isinstance(value, (int, float)) and value is not None:
                    gt_time_fields.append((key, value))
            
            # 如果有时间字段和预测数字，计算相对误差
            if gt_time_fields and predicted_times:
                # 按顺序匹配：第一个gt时间对应第一个预测数字，以此类推
                for i, (field_name, gt_time) in enumerate(gt_time_fields):
                    if i < len(predicted_times):
                        predicted_time = predicted_times[i]
                        # 计算相对误差
                        relative_error = abs(predicted_time - gt_time) / abs(gt_time)
                        relative_errors.append(relative_error)
    
    # 计算结果
    results = {}
    results['accuracy'] = correct_detection / total_samples if total_samples > 0 else 0.0
    
    # 计算精确率、召回率和F1分数
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results['f1_score'] = f1_score

    # 成功率：非None预测占比
    results['success_rate'] = success_count / total_samples if total_samples > 0 else 0.0

    if task == "Event Detection":
        results['mean_relative_error'] = np.mean(relative_errors) if relative_errors else 0.0
        results['median_relative_error'] = np.median(relative_errors) if relative_errors else 0.0
    
    return results


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSONL文件加载数据
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        数据列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


def save_results_to_csv(results: List[Dict[str, Any]], output_csv: str):
    """
    将评估结果保存到CSV文件
    
    Args:
        results: 评估结果列表，每个元素是一个字典，包含各个指标
        output_csv: 输出CSV文件路径
    """
    if not results:
        print("No results to save.")
        return
    
    # 根据filename进行排序
    results.sort(key=lambda x: x['filename'])
    
    # 获取所有可能的列名
    all_columns = set()
    for result in results:
        all_columns.update(result.keys())
    
    # 确保filename列在第一位
    columns = ['filename'] + sorted([col for col in all_columns if col != 'filename'])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Processed {len(results)} files")


def evaluate_all_files(input_folder: str, output_csv: str):
    """
    遍历文件夹中的所有JSONL文件，根据数据集类型调用相应的评估函数
    
    Args:
        input_folder: 输入文件夹路径
        output_csv: 输出CSV文件路径
    """
    results = []
    results_suffix = []
    
    # 遍历文件夹中的所有JSONL文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            
            # 加载数据
            data_list = load_jsonl(file_path)
            if not data_list:
                print(f"  Warning: No data found in {filename}")
                continue
            
            # 获取第一个样本的信息来判断数据集类型
            first_item = data_list[0]
            dataset_name = first_item.get('dataset_name', '')
            task = first_item.get('task', '')
            
            # 根据数据集类型调用相应的评估函数
            result_row = {'filename': filename}

            if task == "Classification" or task == "QA":
                print(f"  Evaluating as classification task: {task}")
                eval_results = classification_eval(data_list)                
            else:
                print(f"  Evaluating as detection task (task: {task})...")
                eval_results = detection_eval(data_list, dataset_name, task)

            for metric_key, value in eval_results.items():
                result_row[metric_key] = f"{(value * 100):.2f}"  # 转为百分数并保留两位小数
            
            if "accuracy" in result_row:
                results.append(result_row)
            else:
                results_suffix.append(result_row)
    
    # 保存结果到CSV
    if output_csv:
        save_results_to_csv(results, output_csv)
        if results_suffix:
            base, ext = os.path.splitext(output_csv)
            output_csv_suffix = f"{base}_suffix{ext}"
            save_results_to_csv(results_suffix, output_csv_suffix)
    else:
        print("No output CSV file specified. Skipping saving results.")
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate benchmark results from JSONL files')
    parser.add_argument('input_folder', help='Input folder containing JSONL files')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)
    
    # 运行评估
    evaluate_all_files(args.input_folder, args.output)
