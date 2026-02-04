import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from accelerate import Accelerator

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'Updating learning rate to {lr}')
            else:
                print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, accelerator: Accelerator=None, patience=7, verbose=False, delta=0):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_error_batch(batch_file_path):
    """
    Load and inspect an error batch file saved during training
    
    Args:
        batch_file_path (str): Path to the .pt file containing the error batch
        
    Returns:
        dict: The loaded batch data containing error information and batch data
    """
    try:
        batch_data = torch.load(batch_file_path, map_location='cpu')
        
        print(f"Error Batch Information:")
        print(f"========================")
        print(f"Epoch: {batch_data.get('epoch', 'N/A')}")
        print(f"Iteration: {batch_data.get('iter', 'N/A')}")
        print(f"Process Index: {batch_data.get('process_index', 'N/A')}")
        print(f"Total Processes: {batch_data.get('num_processes', 'N/A')}")
        print(f"Timestamp: {batch_data.get('timestamp', 'N/A')}")
        print(f"Error Message: {batch_data.get('error_message', 'N/A')}")
        print()
        
        if 'batch_forecast' in batch_data and batch_data['batch_forecast'] is not None:
            print("Forecast Batch Contents:")
            for key, value in batch_data['batch_forecast'].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            print()
        
        if 'analysis_batch' in batch_data and batch_data['analysis_batch'] is not None:
            print("Analysis Batch Contents:")
            for key, value in batch_data['analysis_batch'].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            print()
        
        return batch_data
        
    except Exception as e:
        print(f"Failed to load error batch: {e}")
        return None


def find_all_error_batches(error_batches_dir):
    """
    Find and list all error batch files from all processes
    
    Args:
        error_batches_dir (str): Path to the error_batches directory
        
    Returns:
        dict: Dictionary with process index as key and list of error files as value
    """
    import glob
    
    if not os.path.exists(error_batches_dir):
        print(f"Error batches directory not found: {error_batches_dir}")
        return {}
    
    error_files = {}
    
    # Check for single-process error files (no process subdirectory)
    single_process_files = glob.glob(os.path.join(error_batches_dir, "error_batch_*.pt"))
    if single_process_files:
        error_files['single_process'] = single_process_files
    
    # Check for multi-process error files (in process_X subdirectories)
    process_dirs = glob.glob(os.path.join(error_batches_dir, "process_*"))
    for process_dir in process_dirs:
        process_index = os.path.basename(process_dir)
        process_error_files = glob.glob(os.path.join(process_dir, "error_batch_*.pt"))
        if process_error_files:
            error_files[process_index] = process_error_files
    
    return error_files


def load_all_error_batches(error_batches_dir):
    """
    Load and inspect all error batch files from all processes
    
    Args:
        error_batches_dir (str): Path to the error_batches directory
    """
    error_files = find_all_error_batches(error_batches_dir)
    
    if not error_files:
        print("No error batch files found.")
        return
    
    print(f"Found error batches from {len(error_files)} process(es):")
    print("=" * 60)
    
    for process_key, files in error_files.items():
        print(f"\n{process_key.upper()}:")
        print("-" * 40)
        for file_path in sorted(files):
            print(f"  {os.path.basename(file_path)}")
            
            # Load and show brief info
            try:
                batch_data = torch.load(file_path, map_location='cpu')
                epoch = batch_data.get('epoch', 'N/A')
                iter_idx = batch_data.get('iter', 'N/A')
                process_idx = batch_data.get('process_index', 'N/A')
                error_msg = batch_data.get('error_message', 'N/A')
                print(f"    Epoch: {epoch}, Iter: {iter_idx}, Process: {process_idx}")
                print(f"    Error: {error_msg[:100]}{'...' if len(str(error_msg)) > 100 else ''}")
                print()
            except Exception as e:
                print(f"    Failed to load: {e}")
                print()


# Usage examples for loading error batches:
# 
# 1. Load a single error batch file:
# from run_main_refactored_unified import load_error_batch
# batch_data = load_error_batch('/path/to/error_batch_epoch1_iter100_20231201_123456.pt')
#
# 2. Find and list all error batches from all processes:
# from run_main_refactored_unified import find_all_error_batches, load_all_error_batches
# error_files = find_all_error_batches('/path/to/exp/setting/timestamp/error_batches')
# load_all_error_batches('/path/to/exp/setting/timestamp/error_batches')