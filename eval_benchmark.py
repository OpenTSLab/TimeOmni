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
    "TS_MQA": {"false": ['normal','no anomalies'], "true": ['anomaly','anomalous']},
}


def classification_eval(data_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate accuracy, UAR, and F1 for classification and QA tasks.
    Supports single-label and multi-label classification.

    Args:
        data_list: List of samples loaded from JSONL. Each item contains:
                  - generated_text: model-generated text
                  - gt_result: ground truth info with gt_class or answer

    Returns:
        Dict with accuracy, UAR, and F1 metrics for each key.
    """
    if not data_list:
        return {}
    
    # Check the data format to determine classification vs QA task.
    first_item = data_list[0]
    gt_result = first_item['gt_result']
    
    # Unify handling: convert QA format to classification format.
    if 'answer' in gt_result:
        # QA format: {"answer": "B"} -> {"gt_class": {"answer": "B"}}
        all_keys = ['answer']
        is_qa_task = True
    elif 'gt_class' in gt_result:
        # Classification format: {"gt_class": {"default": ["MI", "CD"]}}
        all_keys = gt_result['gt_class'].keys()
        is_qa_task = False
    else:
        return {}

    # Initialize counters.
    correct_counts = {key: 0 for key in all_keys}
    total_counts = {key: 0 for key in all_keys}

    # Collect all possible class labels.
    all_labels = {}
    for key in all_keys:
        all_labels[key] = set()
    
    # First pass: collect all labels.
    for item in data_list:
        gt_result = item['gt_result']
        
        # Unified handling for QA and classification formats.
        if is_qa_task and 'answer' in gt_result:
            # QA task: map answer to gt_class format.
            unified_gt_class = {'answer': gt_result['answer']}
        elif 'gt_class' in gt_result:
            # Classification task: use gt_class directly.
            unified_gt_class = gt_result['gt_class']
        else:
            continue
            
        for key, gt_value in unified_gt_class.items():
            if isinstance(gt_value, list):
                all_labels[key].update([label.lower() for label in gt_value])
            else:
                all_labels[key].add(gt_value.lower())
    
    # Initialize TP/FN/FP counters per class.
    class_tp = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    class_fn = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    class_fp = {key: {label: 0 for label in all_labels[key]} for key in all_keys}
    
    # Iterate over samples.
    for item in data_list:
        generated_text = item.get('generated_text', '').lower()  # Lowercase for matching.
        gt_result = item['gt_result']
        
        # Unified handling for QA and classification formats.
        if is_qa_task and 'answer' in gt_result:
            # QA task: map answer to gt_class format.
            unified_gt_class = {'answer': gt_result['answer']}
        elif 'gt_class' in gt_result:
            # Classification task: use gt_class directly.
            unified_gt_class = gt_result['gt_class']
        else:
            continue
        
        # Unified evaluation logic.
        for key, gt_value in unified_gt_class.items():
            total_counts[key] += 1
            
            # Handle single-label vs multi-label.
            if isinstance(gt_value, list):
                # Multi-label: compute partial score, UAR, and F1.
                correct_labels = 0
                total_labels = len(gt_value)
                
                # Collect predicted positive labels.
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
                
                # False positives: predicted positive but actually negative.
                for label in predicted_labels:
                    if label not in gt_labels_set:
                        class_fp[key][label] += 1
                
                # Partial credit: correct labels / total labels.
                if total_labels > 0:
                    partial_score = correct_labels / total_labels
                    correct_counts[key] += partial_score
            else:
                # Single-label: check if ground truth appears in generated text.
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
                
                # False positives.
                for label in predicted_labels:
                    if label != gt_label_lower:
                        class_fp[key][label] += 1
    
    # Compute accuracy, UAR, and F1.
    results = {}
    for key in all_keys:
        # Determine metric name suffix.
        if key in ['default', 'answer']:
            suffix = ''
        else:
            suffix = f'_{key}'
        
        if total_counts[key] > 0:
            results[f'accuracy{suffix}'] = correct_counts[key] / total_counts[key]
        else:
            results[f'accuracy{suffix}'] = 0.0
        
        # Unified UAR and F1 computation.
        recalls = []
        precisions = []
        f1_scores = []
        
        for label in all_labels[key]:
            tp = class_tp[key][label]
            fn = class_fn[key][label]
            fp = class_fp[key][label]
            
            # Recall.
            if tp + fn > 0:
                recall = tp / (tp + fn)
                recalls.append(recall)
            
            # Precision.
            if tp + fp > 0:
                precision = tp / (tp + fp)
                precisions.append(precision)
            
            # F1 score.
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
    Detect event results in the generated text using DETECTION_DICT.

    Logic:
    1. If any false indicator appears, return False.
    2. If the true list is empty and no false indicator is found, return True.
    3. If the true list is not empty, check for any true indicator.
    4. If neither is found, return None.
    """
    if dataset_name not in DETECTION_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DETECTION_DICT")
    
    detection_config = DETECTION_DICT[dataset_name]
    false_indicators = detection_config.get("false", [])
    true_indicators = detection_config.get("true", [])
    
    generated_lower = generated_text.lower()

    # import pdb; pdb.set_trace()
    
    # 1. Any false indicator -> False.
    for indicator in false_indicators:
        if indicator.lower() in generated_lower:
            return False
    
    # 2. Empty true list and no false indicator -> True.
    if not true_indicators:
        return True
    
    # 3. Non-empty true list: check indicators.
    for indicator in true_indicators:
        if indicator.lower() in generated_lower:
            return True
    
    # 4. Neither found -> None.
    return None


def extract_predicted_times(generated_text: str) -> List[int]:
    """
    Extract all predicted time values from generated text.

    Args:
        generated_text: model-generated text

    Returns:
        List[int]: extracted time values in order of appearance
    """
    # Extract numbers via regex.
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        return [int(num) for num in numbers]
    return []


def detection_eval(data_list: List[Dict[str, Any]], dataset_name: str, task: str) -> Dict[str, float]:
    """
    Evaluate detection task accuracy, F1, and relative error.

    Args:
        data_list: List of samples loaded from JSONL. Each item contains:
                  - generated_text: model-generated text
                  - gt_result: ground truth with contain and time fields
        dataset_name: dataset name used to select detection dictionary
        task: task type, "Event Detection" or "Anomaly Detection"

    Returns:
        Dict with detection accuracy, F1 score, and median relative error
    """
    if not data_list:
        return {}
    
    correct_detection = 0
    total_samples = 0
    relative_errors = []
    
    # Confusion-matrix stats for F1.
    true_positive = 0   # True positive: actual positive, predicted positive
    false_positive = 0  # False positive: actual negative, predicted positive
    true_negative = 0   # True negative: actual negative, predicted negative
    false_negative = 0  # False negative: actual positive, predicted negative
    
    # Success rate: proportion of non-None predictions.
    success_count = 0
    
    for item in data_list:
        generated_text = item.get('generated_text', '')
        
        if 'gt_result' not in item:
            continue
            
        gt_result = item['gt_result']
        gt_contain = gt_result.get('contain', False)
        
        total_samples += 1
        
        predicted_contain = detect(item['generated_text'], dataset_name)
        
        # If undecidable, count as failure in denominator only.
        if predicted_contain is None:
            # Exclude from confusion matrix and relative error.
            continue

        # Decidable prediction counts as success.
        success_count += 1
        
        detection_correct = (predicted_contain == gt_contain)
        if detection_correct:
            correct_detection += 1
        
        # Update confusion matrix for F1.
        if gt_contain == True and predicted_contain == True:
            true_positive += 1
        elif gt_contain == False and predicted_contain == True:
            false_positive += 1
        elif gt_contain == False and predicted_contain == False:
            true_negative += 1
        elif gt_contain == True and predicted_contain == False:
            false_negative += 1
            
        # For event detection, extract numbers only when detection is correct and positive.
        if task == "event detection" and gt_contain and predicted_contain:
            predicted_times = extract_predicted_times(generated_text)
            
            gt_time_fields = []
            for key, value in gt_result.items():
                if key != 'contain' and isinstance(value, (int, float)) and value is not None:
                    gt_time_fields.append((key, value))
            
            # If time fields and predicted numbers exist, compute relative error.
            if gt_time_fields and predicted_times:
                # Match by order: first GT time with first predicted number, etc.
                for i, (field_name, gt_time) in enumerate(gt_time_fields):
                    if i < len(predicted_times):
                        predicted_time = predicted_times[i]
                        # Compute relative error.
                        relative_error = abs(predicted_time - gt_time) / abs(gt_time)
                        relative_errors.append(relative_error)
    
    # Compute results.
    results = {}
    results['accuracy'] = correct_detection / total_samples if total_samples > 0 else 0.0
    
    # Compute precision, recall, and F1.
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results['f1_score'] = f1_score

    # Success rate: proportion of non-None predictions.
    results['success_rate'] = success_count / total_samples if total_samples > 0 else 0.0

    if task == "event detection":
        results['mean_relative_error'] = np.mean(relative_errors) if relative_errors else 0.0
        results['median_relative_error'] = np.median(relative_errors) if relative_errors else 0.0
    
    return results


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: JSONL file path

    Returns:
        List of data records
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
    Save evaluation results to a CSV file.

    Args:
        results: List of result dicts with metrics
        output_csv: Output CSV path
    """
    if not results:
        print("No results to save.")
        return
    
    # Sort by filename.
    results.sort(key=lambda x: x['filename'])
    
    # Collect all possible column names.
    all_columns = set()
    for result in results:
        all_columns.update(result.keys())
    
    # Ensure filename is the first column.
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
    Traverse JSONL files in a folder and run evaluation by dataset type.

    Args:
        input_folder: input folder path
        output_csv: output CSV path
    """
    results = []
    results_suffix = []
    
    # Iterate over JSONL files in the folder.
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            
            # Load data.
            data_list = load_jsonl(file_path)
            if not data_list:
                print(f"  Warning: No data found in {filename}")
                continue
            
            # Use the first sample to determine dataset type.
            first_item = data_list[0]
            dataset_name = first_item.get('dataset_name', '')
            task = first_item.get('task', '').lower()

            # Call the appropriate evaluation based on dataset type.
            result_row = {'filename': filename}

            if task == "classification" or task == "qa":
                print(f"  Evaluating as classification task: {task}")
                eval_results = classification_eval(data_list)                
            else:
                print(f"  Evaluating as detection task (task: {task})...")
                eval_results = detection_eval(data_list, dataset_name, task)

            for metric_key, value in eval_results.items():
                result_row[metric_key] = f"{(value * 100):.2f}"  # Percent with 2 decimals.
            
            if "accuracy" in result_row:
                results.append(result_row)
            else:
                results_suffix.append(result_row)
    
    # Save results to CSV.
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
    
    # Check input folder existence.
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)
    
    # Run evaluation.
    evaluate_all_files(args.input_folder, args.output)