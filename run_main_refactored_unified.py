import argparse
import time
import random
import os
import json
import sys
import subprocess
import shutil
import traceback
import csv
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Import PEFT for DoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. DoRA fine-tuning will not work. Install with: pip install peft")
    PEFT_AVAILABLE = False

from models import TimeOmni
from data_provider.data_factory_unified import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate


class ConfigManager:
    """Manages configuration and environment information collection and saving"""
    
    # 重要程序文件清单
    IMPORTANT_FILES = [
        # 主要运行脚本
        'run_main_refactored_unified.py',
        
        # 模型文件
        'models/TimeOmni.py',
        
        # 数据提供者
        'data_provider/data_factory_unified.py',
        'data_provider/dataset.py',

        # tools
        'utils/tools.py',
        'infer_benchmark.py',
        'eval_benchmark.py',

        # scripts
        'scripts/TimeOmni_unified.sh',
        'scripts/TimeOmni_unified_single.sh'
    ]
    
    @staticmethod
    def save_important_code_files(path: str, accelerator: Accelerator) -> None:
        """Save important code files to log folder"""
        try:
            # 创建代码备份文件夹
            code_backup_dir = os.path.join(path, 'code_backup')
            if not os.path.exists(code_backup_dir):
                os.makedirs(code_backup_dir)
            
            saved_files = []
            missing_files = []
            
            # 获取项目根目录
            current_dir = os.getcwd()
            
            for file_path in ConfigManager.IMPORTANT_FILES:
                full_path = os.path.join(current_dir, file_path)
                
                if os.path.exists(full_path):
                    try:
                        # 创建目标目录结构
                        target_file_path = os.path.join(code_backup_dir, file_path)
                        target_dir = os.path.dirname(target_file_path)
                        
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        
                        # 复制文件
                        shutil.copy2(full_path, target_file_path)
                        saved_files.append(file_path)
                        
                    except Exception as e:
                        accelerator.print(f"Warning: Could not copy {file_path}: {e}")
                        missing_files.append(f"{file_path} (copy error: {e})")
                else:
                    missing_files.append(f"{file_path} (not found)")
            
            accelerator.print(f"Code backup completed: {len(saved_files)}/{len(ConfigManager.IMPORTANT_FILES)} files saved to {code_backup_dir}")
            if missing_files:
                accelerator.print(f"Warning: {len(missing_files)} files could not be backed up")
                
        except Exception as e:
            accelerator.print(f"Error: Could not save important code files: {e}")
    
    @staticmethod
    def save_conda_env_info(path: str, accelerator: Accelerator) -> None:
        """Save conda environment information"""
        try:
            conda_env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
            conda_list_result = subprocess.run(['conda', 'list'], 
                                             capture_output=True, text=True, timeout=30)
            
            conda_file = os.path.join(path, 'conda_environment.txt')
            with open(conda_file, 'w', encoding='utf-8') as f:
                f.write(f"Conda Environment: {conda_env_name}\n")
                f.write("="*50 + "\n")
                
                if conda_list_result.returncode == 0:
                    f.write(conda_list_result.stdout)
                else:
                    f.write("Error: conda list command failed\n")
                    
        except Exception as e:
            accelerator.print(f"Warning: Could not save conda environment info: {e}")
    
    @staticmethod
    def get_system_info(accelerator: Accelerator) -> Dict[str, Any]:
        """Get system information"""
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': str(accelerator.device),
            'num_processes': accelerator.num_processes,
            'mixed_precision': str(accelerator.mixed_precision) if hasattr(accelerator, 'mixed_precision') else None
        }
    
    @staticmethod
    def save_pip_requirements(path: str, accelerator: Accelerator) -> None:
        """Save pip freeze requirements"""
        try:
            pip_freeze_result = subprocess.run(['pip', 'freeze'], 
                                             capture_output=True, text=True, timeout=30)
            if pip_freeze_result.returncode == 0:
                requirements_file = os.path.join(path, 'requirements.txt')
                with open(requirements_file, 'w', encoding='utf-8') as f:
                    f.write(pip_freeze_result.stdout)
        except Exception as e:
            accelerator.print(f"Warning: Could not save pip freeze requirements: {e}")
    
    @staticmethod
    def save_config(args_dict, path: str, accelerator: Accelerator) -> None:
        """Save configuration and environment information"""
        # Add system and environment information
        args_dict['system_info'] = ConfigManager.get_system_info(accelerator)
        
        # Save conda environment to separate text file
        ConfigManager.save_conda_env_info(path, accelerator)
        
        # Save pip requirements
        ConfigManager.save_pip_requirements(path, accelerator)
        
        # Save important code files backup
        ConfigManager.save_important_code_files(path, accelerator)
        
        # Save main config file
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)


class DualLogger:
    """Custom logger that writes to both console and file"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message: str):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
            
    def flush(self):
        self.terminal.flush()


class TrainingLogger:
    """Manages training logging and experiment tracking"""
    
    def __init__(self, path: str, accelerator: Accelerator):
        self.path = path
        self.accelerator = accelerator
        self.original_stdout = None

    def setup_logging(self, args_dict) -> None:
        """Setup dual logging and save configuration"""
        if self.accelerator.is_main_process:
            # Setup dual logger (no timestamp suffix since we're already in a timestamped folder)
            log_file = os.path.join(self.path, 'training_log.txt')
            self.original_stdout = sys.stdout
            sys.stdout = DualLogger(log_file)
            ConfigManager.save_config(args_dict, self.path, self.accelerator)
    
    def save_training_summary(self, setting: str, epoch: int, train_loss: float, 
                            vali_loss: float, test_loss: float, test_mae_loss: float,
                            early_stopping, args, **kwargs) -> None:
        """Save final training summary"""
        if self.accelerator.is_main_process:
            summary_file = os.path.join(self.path, 'training_summary.json')
            training_summary = {
                'experiment_setting': setting,
                'total_epochs_trained': epoch + 1,
                'final_train_loss': float(train_loss),
                'final_vali_loss': float(vali_loss),
                'final_test_loss': float(test_loss),
                'final_mae_loss': float(test_mae_loss),
                'early_stopped': early_stopping.early_stop if early_stopping else False,
                'training_completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_model_path': self.path
            }
            
            # Add analysis losses if available
            if (args.add_analysis or args.analysis_only) and kwargs:
                training_summary['final_predict_loss'] = kwargs.get('predict_loss_avg')
                training_summary['final_analysis_loss'] = kwargs.get('analysis_loss_avg')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=2, ensure_ascii=False)
            
            self.accelerator.print("="*80)
            self.accelerator.print(f"Training completed! Summary saved to: {summary_file}")
            self.accelerator.print("="*80)
    
    def restore_stdout(self) -> None:
        """Restore original stdout"""
        if self.accelerator.is_main_process and self.original_stdout:
            sys.stdout = self.original_stdout


class Trainer:
    """Unified trainer for both forecast and analysis tasks"""

    def __init__(self, args, model, accelerator: Accelerator):
        self.args = args
        self.model = model
        self.accelerator = accelerator
        self.dual_task = args.add_analysis
        self.analysis_only = args.analysis_only
    
    def train_step(self, batch_forecast, analysis_batch=None, scaler=None):
        """Single training step"""
        if self.args.use_amp and scaler:
            with torch.cuda.amp.autocast():
                return self._forward_computation(batch_forecast, analysis_batch)
        else:
            return self._forward_computation(batch_forecast, analysis_batch)
    
    def _forward_computation(self, batch_forecast, analysis_data):
        """Forward computation for different training modes"""
        if self.analysis_only:
            return self._analysis_forward_pass(analysis_data)
        
        predict_loss = self._forecast_forward_pass(batch_forecast)

        if self.dual_task and analysis_data is not None:
            analysis_loss = self._analysis_forward_pass(analysis_data)
            total_loss = predict_loss + self.args.analysis_loss_weight * analysis_loss
            return total_loss, predict_loss.item(), analysis_loss.item()
        else:
            return predict_loss

    def _forecast_forward_pass(self, batch_forecast):
        """Forecast forward pass"""
        device = self.accelerator.device
        input_ts = batch_forecast.get('input_ts_list')
        input_ts = [input.to(self.accelerator.device) for input in input_ts]
        batch_y_list = batch_forecast.get('gt_ts_list', None) # list[tensor(shape(T, C))]
        input_ts_mask_list = batch_forecast.get('input_ts_mask_list', None) # list[tensor(shape(T, C)) | None]
        input_ids = batch_forecast['input_ids'].to(self.accelerator.device)

        outputs, _ = self.model(input_ts, input_ids=input_ids, gt_ts=batch_y_list, mode='forecast')

        # Compute loss using a list of GT tensors by aligning the last part of outputs
        predict_loss = None

        losses = []
        for idx, gt in enumerate(batch_y_list):
            if gt is None:
                continue
            gt = gt.float().to(device)

            # Align the last pred steps to the gt length
            pred_len = gt.size(0)
            pred = outputs[idx][-pred_len:, :]

            losses.append(F.mse_loss(pred, gt))

        if len(losses) == 0:
            # No valid GTs; fall back to zero to avoid crashing
            predict_loss = torch.zeros((), device=device, dtype=outputs.dtype)
        else:
            predict_loss = torch.stack(losses).mean()

        # Free temps
        del outputs, input_ts, input_ids, batch_y_list, losses, gt, pred

        return predict_loss
    
    def _analysis_forward_pass(self, analysis_data):
        """Analysis forward pass"""
        # Use raw list of variable-length tensors so model can choose patch embeddings per sample
        input_ts = analysis_data.get('input_ts_list')
        input_ts = [input.to(self.accelerator.device) for input in input_ts]
        input_ids = analysis_data['input_ids'].to(self.accelerator.device)
        labels = analysis_data['labels'].to(self.accelerator.device)

        analysis_outputs = self.model(input_ts, input_ids=input_ids, mode='analyze')

        logits = analysis_outputs.logits
        logits = logits[:, -input_ids.size(1):, :]
        
        if len(logits.shape) == 3 and logits.size(1) > 1:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
        else:
            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
        
        analysis_loss = F.cross_entropy(shift_logits, shift_labels)

        del input_ts, input_ids, labels, logits, shift_logits, shift_labels, analysis_outputs
        
        return analysis_loss


class TimeOmniExperiment:
    """Main experiment class that orchestrates the entire training process"""
    
    def __init__(self, args, experiment_id: int = 0):
        self.args = args

        # Setup accelerator with distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

        # Setup paths with timestamp folder
        self.setting = f"lr{self.args.learning_rate}_{self.args.lradj}_bs{self.args.batch_size}_epochs{self.args.train_epochs}"
        self.path = os.path.join(self.args.exp_dir, self.args.model_comment, self.setting, f'exp{experiment_id}')
        if self.accelerator.is_main_process:
            os.makedirs(self.path, exist_ok=True)
            # Create forecast_results folder for validation results
            self.forecast_results_path = os.path.join(self.path, args.forecast_results_dir)
            os.makedirs(self.forecast_results_path, exist_ok=True)
        else:
            self.forecast_results_path = os.path.join(self.path, args.forecast_results_dir)
        self.keep_epochs = set(range(self.args.train_epochs, 0, -max(1, self.args.train_epochs // self.args.max_keep_epochs)))

        # Setup logging
        self.logger = TrainingLogger(self.path, self.accelerator)
        self.logger.setup_logging(self.args_dict)
        
        # Create model and tokenizer
        if self.args.model == 'TimeOmni':
            self.model = TimeOmni.Model(self.args).float()
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")
        
        self.vali_data, self.vali_loader, _, _ = data_provider(self.args, 'val', self.tokenizer)
        
        # Support multiple test datasets
        if hasattr(self.args, 'forecast_test_jsonl_file_path') and self.args.forecast_test_jsonl_file_path:
            if isinstance(self.args.forecast_test_jsonl_file_path, list):
                self.test_datasets = []
                self.test_loaders = []
                self.test_names = []
                forecast_test_jsonl_file_paths = self.args.forecast_test_jsonl_file_path
                for i, test_path in enumerate(forecast_test_jsonl_file_paths):
                    # Temporarily set the single test path for data_provider
                    self.args.forecast_test_jsonl_file_path = test_path
                    test_data, test_loader, _, _ = data_provider(self.args, 'test', self.tokenizer)
                    self.test_datasets.append(test_data)
                    self.test_loaders.append(test_loader)
                    # Extract test name from file path
                    test_name = test_path.split('/')[-1].replace('.jsonl', '')
                    self.test_names.append(test_name)
            else:
                # Single test dataset (backward compatibility)
                test_data, test_loader, _, _ = data_provider(self.args, 'test', self.tokenizer)
                self.test_datasets = [test_data]
                self.test_loaders = [test_loader]
                test_name = self.args.forecast_test_jsonl_file_path.split('/')[-1].replace('.jsonl', '')
                self.test_names = [test_name]
        else:
            # No test dataset specified
            self.test_datasets = []
            self.test_loaders = []
            self.test_names = []
        
        self.train_data, self.train_loader, self.train_analysis_data, self.train_analysis_loader = data_provider(self.args, 'train', self.tokenizer)
        
        if self.args.analysis_only:
            self.accelerator.print(f"Analysis-only mode: Analysis dataset len: {len(self.train_analysis_data) if self.train_analysis_data else 'N/A'}")
        else:
            self.accelerator.print(f"Forecast dataset len: {len(self.train_data) if self.train_data else 'N/A'}, Analysis dataset len: {len(self.train_analysis_data) if self.train_analysis_data else 'N/A'}")
            
        if self.train_analysis_loader:
            self.accelerator.print(f"Analysis dataloader batch size: {self.train_analysis_loader.batch_size}")
        if not self.args.analysis_only and self.train_loader:
            self.accelerator.print(f"Forecast dataloader batch size: {self.train_loader.batch_size}")
        
        # Setup optimizer
        trained_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model_optim = optim.Adam(trained_parameters, lr=self.args.learning_rate)
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if self.args.lradj != "COSINE_WARMUP":
            self.setup_scheduler()

        if self.train_analysis_loader is not None:
            # Prepare all test loaders
            prepared_items = [self.train_loader, self.vali_loader] + self.test_loaders + [self.train_analysis_loader, self.model, self.model_optim]
            prepared_results = self.accelerator.prepare(*prepared_items)
            self.train_loader = prepared_results[0]
            self.vali_loader = prepared_results[1]
            self.test_loaders = prepared_results[2:2+len(self.test_loaders)]
            self.train_analysis_loader = prepared_results[2+len(self.test_loaders)]
            self.model = prepared_results[2+len(self.test_loaders)+1]
            self.model_optim = prepared_results[2+len(self.test_loaders)+2]
        else:
            # Prepare all test loaders
            prepared_items = [self.train_loader, self.vali_loader] + self.test_loaders + [self.model, self.model_optim]
            prepared_results = self.accelerator.prepare(*prepared_items)
            self.train_loader = prepared_results[0]
            self.vali_loader = prepared_results[1]
            self.test_loaders = prepared_results[2:2+len(self.test_loaders)]
            self.model = prepared_results[2+len(self.test_loaders)]
            self.model_optim = prepared_results[2+len(self.test_loaders)+1]

        if self.args.lradj != 'COSINE_WARMUP':
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Get correct train_steps after preparation
        if self.args.analysis_only or self.args.add_analysis:
            # Use analysis dataloader as the main driver for epoch calculation
            self.train_steps = len(self.train_analysis_loader)
            if self.args.analysis_only:
                self.accelerator.print(f"Analysis-only mode: Analysis dataloader len: {self.train_steps}")
            else:
                self.accelerator.print(f"Dual-task mode (analysis-driven): Analysis dataloader len: {self.train_steps}, Forecast dataloader len: {len(self.train_loader) if self.train_loader else 'N/A'}")
        else:
            # Forecast-only mode
            self.train_steps = len(self.train_loader)
            self.accelerator.print(f"Forecast-only mode: Forecast dataloader len: {self.train_steps}")

        if self.args.lradj == "COSINE_WARMUP":
            self.setup_scheduler()

        # Load checkpoint if provided
        if self.args.ckpt_path and self.args.load_training_states:
            self.accelerator.load_state(self.args.ckpt_path, load_module_strict=False)
        elif self.args.ckpt_path and not self.args.load_training_states:
            self.accelerator.load_state(self.args.ckpt_path, load_module_strict=False, load_module_only=True)
        
        # # Setup loss functions
        # self.criterion = nn.MSELoss()
        # self.mae_metric = nn.L1Loss()
        # self.ce_criterion = nn.CrossEntropyLoss()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience) if self.args.use_early_stop else None
        
        # Setup trainer
        self.trainer = Trainer(self.args, self.model, self.accelerator)

    @property
    def args_dict(self):
        config_dict = vars(self.args).copy()
        
        # Convert non-serializable objects to strings
        for key, value in config_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = str(value)

        return config_dict

    def setup_scheduler(self):
        if self.args.lradj == 'COS':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.model_optim, T_max=20, eta_min=1e-8)
        elif self.args.lradj == 'COSINE_WARMUP':
            total_steps = self.train_steps * self.args.train_epochs
            
            if self.args.warmup_steps:
                num_warmup_steps = self.args.warmup_steps
            else:
                num_warmup_steps = int(total_steps * self.args.warmup_ratio)
            
            self.accelerator.print(f"Total training steps: {total_steps}, using {num_warmup_steps} warmup steps for cosine schedule with warmup.")
            
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.model_optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
            self.scheduler = self.accelerator.prepare(self.scheduler)
        else:
            self.scheduler = lr_scheduler.OneCycleLR(
                optimizer=self.model_optim,
                steps_per_epoch=self.train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )

    def save_checkpoint(self, epoch):
        try:
            checkpoint_dir = os.path.join(self.path, f'epoch_{epoch}')
            self.accelerator.save_state(checkpoint_dir, exclude_frozen_parameters=True)

            if self.accelerator.is_main_process:
                self._clean_old_checkpoints(epoch)
        except Exception as e:
            self.accelerator.print(f"Warning: Could not save checkpoint for epoch {epoch}: {e}")
        
    def _clean_old_checkpoints(self, epoch):
        if (epoch - 1) not in self.keep_epochs:
            old_checkpoint_dir = os.path.join(self.path, f'epoch_{epoch - 1}')
            if os.path.exists(old_checkpoint_dir) and self.accelerator.is_main_process:
                try:
                    shutil.rmtree(old_checkpoint_dir)
                    self.accelerator.print(f"Removed old checkpoint directory: epoch_{epoch - 1}")
                except OSError as e:
                    self.accelerator.print(f"Warning: Failed to remove {old_checkpoint_dir}: {e}")
    
    def save_error_batch(self, batch_forecast, analysis_batch, epoch, iter_idx, error_msg):
        """Save current batch data when training error occurs"""
        try:
            # Get process information
            process_index = self.accelerator.process_index
            num_processes = self.accelerator.num_processes
            
            # Create error directory with process-specific subdirectory
            error_dir = os.path.join(self.path, 'error_batches')
            if num_processes > 1:
                process_error_dir = os.path.join(error_dir, f'process_{process_index}')
            else:
                process_error_dir = error_dir
                
            os.makedirs(process_error_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if num_processes > 1:
                error_filename = f"error_batch_epoch{epoch}_iter{iter_idx}_process{process_index}_{timestamp}.pt"
            else:
                error_filename = f"error_batch_epoch{epoch}_iter{iter_idx}_{timestamp}.pt"
            error_path = os.path.join(process_error_dir, error_filename)
            
            # Prepare batch data to save
            batch_data = {
                'epoch': epoch,
                'iter': iter_idx,
                'timestamp': timestamp,
                'process_index': process_index,
                'num_processes': num_processes,
                'error_message': str(error_msg),
                'batch_forecast': batch_forecast,
                'analysis_batch': analysis_batch,
                'args': self.args_dict
            }
            
            # Save the batch data
            torch.save(batch_data, error_path)
            
            # Also save error log
            if num_processes > 1:
                error_log_filename = f"error_log_process{process_index}_{timestamp}.txt"
            else:
                error_log_filename = f"error_log_{timestamp}.txt"
            error_log_path = os.path.join(process_error_dir, error_log_filename)
            
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Error Report\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Process Index: {process_index}/{num_processes}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Iteration: {iter_idx}\n")
                f.write(f"Error Message: {error_msg}\n")
                f.write(f"Batch File: {error_filename}\n")
                f.write(f"=" * 50 + "\n")
                
                # Add batch info
                if batch_forecast is not None:
                    f.write(f"Forecast Batch Info:\n")
                    for key, value in batch_forecast.items():
                        if hasattr(value, 'shape'):
                            f.write(f"  {key}: shape {value.shape}, dtype {value.dtype}\n")
                        else:
                            f.write(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else str(value)}\n")
                
                if analysis_batch is not None:
                    f.write(f"Analysis Batch Info:\n")
                    for key, value in analysis_batch.items():
                        if hasattr(value, 'shape'):
                            f.write(f"  {key}: shape {value.shape}, dtype {value.dtype}\n")
                        else:
                            f.write(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else str(value)}\n")
            
            self.accelerator.print(f"Error batch saved to: {error_path}")
            self.accelerator.print(f"Error log saved to: {error_log_path}")
            
        except Exception as save_error:
            self.accelerator.print(f"Failed to save error batch: {save_error}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        train_loss = []
        predict_loss_list = []
        analysis_loss_list = []
        
        self.model.train()
        epoch_time = time.time()
        iter_count = 0
        time_now = time.time()
          
        if self.args.analysis_only or self.args.add_analysis:
            train_forecast_iter = iter(self.train_loader) if self.train_loader and self.args.add_analysis else None
            
            for i, analysis_batch in tqdm(enumerate(self.train_analysis_loader), total=self.train_steps, desc=f'Epoch:{epoch + 1}'):
                try:
                    iter_count += 1
                    self.model_optim.zero_grad()
                    
                    # Get forecast batch if needed for dual-task training
                    forecast_batch = None
                    if train_forecast_iter:
                        try:
                            forecast_batch = next(train_forecast_iter)
                        except StopIteration:
                            train_forecast_iter = iter(self.train_loader)
                            forecast_batch = next(train_forecast_iter)
                    
                    # Training step
                    if self.args.analysis_only:
                        # Analysis-only mode
                        loss = self.trainer.train_step(None, analysis_batch, self.scaler)
                        analysis_loss_list.append(loss.item())
                        train_loss.append(loss.item())
                    else:
                        # Dual-task mode (add_analysis=True)
                        result = self.trainer.train_step(forecast_batch, analysis_batch, self.scaler)
                        if isinstance(result, tuple) and len(result) == 3:
                            # Dual task: (loss, predict_loss_item, analysis_loss_item)
                            loss, predict_loss_item, analysis_loss_item = result
                            predict_loss_list.append(predict_loss_item)
                            analysis_loss_list.append(analysis_loss_item)
                            train_loss.append(loss.item())
                    
                    # Backward pass
                    if self.args.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.model_optim)
                        self.scaler.update()
                    else:
                        self.accelerator.backward(loss)
                        self.model_optim.step()

                    # Update scheduler
                    if self.args.lradj in ['TST', 'COSINE_WARMUP']:
                        if self.args.lradj == 'TST':
                            adjust_learning_rate(self.accelerator, self.model_optim, self.scheduler, epoch + 1, self.args, printout=False)
                        self.scheduler.step()
                    
                    # Print progress
                    if (i + 1) % 100 == 0:
                        if self.args.analysis_only:
                            # Analysis-only progress
                            recent_analysis_avg = np.mean(analysis_loss_list[-100:]) if len(analysis_loss_list) >= 100 else np.mean(analysis_loss_list)
                            self.accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | analysis_loss (avg_100): {recent_analysis_avg:.7f}")
                        elif self.args.add_analysis and predict_loss_list:
                            # Dual-task progress
                            recent_total_avg = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
                            recent_predict_avg = np.mean(predict_loss_list[-100:]) if len(predict_loss_list) >= 100 else np.mean(predict_loss_list)
                            recent_analysis_avg = np.mean(analysis_loss_list[-100:]) if len(analysis_loss_list) >= 100 else np.mean(analysis_loss_list)
                            self.accelerator.print(
                                f"\titers: {i + 1}, epoch: {epoch + 1} | "
                                f"total_loss (avg_100): {recent_total_avg:.7f} | "
                                f"predict_loss (avg_100): {recent_predict_avg:.7f} | "
                                f"analysis_loss (avg_100): {recent_analysis_avg:.7f}"
                            )
                        
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * self.train_steps - i)
                        self.accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                        iter_count = 0
                        time_now = time.time()
                        
                except Exception as e:
                    self.accelerator.print(f"Error occurred at epoch {epoch + 1}, iteration {i + 1}: {e}")
                    # Save error batch
                    self.save_error_batch(forecast_batch, analysis_batch, epoch + 1, i + 1, e)
                    # Re-raise the exception to stop training
                    raise e
        else:
            for i, batch in tqdm(enumerate(self.train_loader), total=self.train_steps, desc=f'Epoch:{epoch + 1}'):
                try:
                    iter_count += 1
                    self.model_optim.zero_grad()
                    
                    # Training step for forecast-only
                    loss = self.trainer.train_step(batch, None, self.scaler)
                    train_loss.append(loss.item())
                    
                    # Backward pass
                    if self.args.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.model_optim)
                        self.scaler.update()
                    else:
                        self.accelerator.backward(loss)
                        self.model_optim.step()
                    
                    # Update scheduler
                    if self.args.lradj in ['TST', 'COSINE_WARMUP']:
                        if self.args.lradj == 'TST':
                            adjust_learning_rate(self.accelerator, self.model_optim, self.scheduler, epoch + 1, self.args, printout=False)
                        self.scheduler.step()
                    
                    # Print progress
                    if (i + 1) % 100 == 0:
                        # Calculate average loss for last 100 iterations
                        recent_avg = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
                        self.accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | forecast_loss (avg_100): {recent_avg:.7f}")
                        
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * self.train_steps - i)
                        self.accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                        iter_count = 0
                        time_now = time.time()
                        
                except Exception as e:
                    print(f"Error occurred at epoch {epoch + 1}, iteration {i + 1}: {e}")
                    # Save error batch
                    self.save_error_batch(batch, None, epoch + 1, i + 1, e)
                    # Re-raise the exception to stop training
                    raise e
        
        self.accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
        
        return train_loss, predict_loss_list, analysis_loss_list
    
    def validate_epoch(self, epoch=-1):
        """Perform validation and testing for one epoch"""
        # Clear CUDA cache and synchronize all processes before validation
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        # Validation
        # vali_loss, vali_mae_loss, valid_loss_norm, valid_mae_loss_norm = self._validate_dataset(self.vali_loader, 'validation', epoch)

        # Testing on multiple test datasets
        test_results = []
        if self.test_loaders:
            for i, (test_loader, test_name) in enumerate(zip(self.test_loaders, self.test_names)):
                test_loss, test_mae_loss, test_loss_norm, test_mae_loss_norm, test_relative_mae, test_relative_mae_norm = self._validate_dataset(
                    test_loader, f'test_{test_name}', epoch
                )
                test_results.append({
                    'name': test_name,
                    'test_loss': test_loss,
                    'test_mae_loss': test_mae_loss,
                    'test_loss_norm': test_loss_norm,
                    'test_mae_loss_norm': test_mae_loss_norm,
                    'test_relative_mae': test_relative_mae,
                    'test_relative_mae_norm': test_relative_mae_norm
                })
        else:
            # No test datasets available
            test_results = []

        # Return validation results and list of test results
        return 0, 0, test_results, 0, 0

    def _validate_dataset(self, dataloader, phase='validation', epoch=-1):
        """Validate on a specific dataset"""
        # Collect per-batch scalar losses locally; we'll aggregate across ranks at the end
        per_batch_loss = []
        per_batch_mae = []
        per_batch_loss_norm = []
        per_batch_mae_norm = []
        per_batch_relative_mae = []
        per_batch_relative_mae_norm = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(dataloader), desc=f"{phase}"):
                try:
                    input_ts = batch_data['input_ts_list']
                    input_ts_mask = batch_data.get('input_ts_mask_list', None)  # List[Tensor(T) | None]
                    # gt_ts = batch_data.get('gt_ts', None)  # (Batch, Time, Channels) 或 None
                    gt_ts_list = batch_data.get('gt_ts_list', None)  # List[Tensor(T, C)] 或 None
                    input_ids = batch_data['input_ids']
                    
                    if input_ts is None:
                        self.accelerator.print(f"Warning: input_ts is None for batch {i}, skipping...")
                        continue

                    input_ts = [input.to(self.accelerator.device) for input in input_ts]

                    # 使用TimeOmni的forecast模式进行预测
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, outputs_norm = self.model(input_ts, input_ids=input_ids, gt_ts=gt_ts_list, mode='forecast')
                    else:
                        outputs, outputs_norm = self.model(input_ts, input_ids=input_ids, gt_ts=gt_ts_list, mode='forecast')

                    sample_losses = []
                    sample_maes = []
                    sample_losses_norm = []
                    sample_maes_norm = []
                    sample_relative_maes = []
                    sample_relative_maes_norm = []
                    for idx, gt in enumerate(gt_ts_list):
                        if gt is None:
                            continue
                        gt = gt.float().to(self.accelerator.device)
                        pred_len = gt.size(0)
                        pred = outputs[idx][-pred_len:, :]
                        pred_norm = outputs_norm[idx][-pred_len:, :]
                        gt_norm = self.model.normalize_layers(gt.unsqueeze(0), 'norm').squeeze(0)

                        input_mask = input_ts_mask[idx]
                        if input_mask is not None:
                            assert input_mask.shape == gt.shape, f"Mask shape {input_mask.shape} does not match GT shape {gt.shape}"

                        # Calculate losses with mask if available
                        if input_mask is not None:
                            # Apply mask to loss calculation
                            mask = input_mask.float().to(self.accelerator.device)
                            
                            mse_loss = F.mse_loss(pred, gt, reduction='none')
                            mae_loss = F.l1_loss(pred, gt, reduction='none')
                            mse_loss_norm = F.mse_loss(pred_norm, gt_norm, reduction='none')
                            mae_loss_norm = F.l1_loss(pred_norm, gt_norm, reduction='none')
                            
                            # Calculate relative MAE: |pred - gt| / |gt| (exclude points where gt == 0)
                            gt_abs = torch.abs(gt)
                            gt_norm_abs = torch.abs(gt_norm)
                            
                            # Create mask for non-zero ground truth values
                            non_zero_mask = (gt_abs > 1e-6).float()
                            non_zero_mask_norm = (gt_norm_abs > 1e-6).float()
                            
                            # Combine with input mask
                            combined_mask = mask * non_zero_mask
                            combined_mask_norm = mask * non_zero_mask_norm
                            
                            # Calculate relative MAE only for non-zero GT values
                            relative_mae = mae_loss / (gt_abs + 1e-6)  # Add small epsilon to avoid division by zero
                            relative_mae_norm = mae_loss_norm / (gt_norm_abs + 1e-6)
                            
                            # Apply combined mask and average over valid positions
                            sample_losses.append((mse_loss * mask).sum() / (mask.sum() + 1e-6))
                            sample_maes.append((mae_loss * mask).sum() / (mask.sum() + 1e-6))
                            sample_losses_norm.append((mse_loss_norm * mask).sum() / (mask.sum() + 1e-6))
                            sample_maes_norm.append((mae_loss_norm * mask).sum() / (mask.sum() + 1e-6))
                            
                            # For relative MAE, only average over positions where gt != 0
                            if combined_mask.sum() > 0:
                                sample_relative_maes.append((relative_mae * combined_mask).sum() / combined_mask.sum())
                            else:
                                sample_relative_maes.append(torch.tensor(0.0, device=self.accelerator.device))
                                
                            if combined_mask_norm.sum() > 0:
                                sample_relative_maes_norm.append((relative_mae_norm * combined_mask_norm).sum() / combined_mask_norm.sum())
                            else:
                                sample_relative_maes_norm.append(torch.tensor(0.0, device=self.accelerator.device))
                        else:
                            # No mask, use standard loss calculation
                            mse_loss = F.mse_loss(pred, gt, reduction='none')
                            mse_loss_norm = F.mse_loss(pred_norm, gt_norm, reduction='none')
                            mae_loss = F.l1_loss(pred, gt, reduction='none')
                            mae_loss_norm = F.l1_loss(pred_norm, gt_norm, reduction='none')
                            sample_losses.append(mse_loss.mean())
                            sample_maes.append(mae_loss.mean())
                            sample_losses_norm.append(mse_loss_norm.mean())
                            sample_maes_norm.append(mae_loss_norm.mean())

                            # Calculate relative MAE for all points
                            gt_abs = torch.abs(gt)
                            gt_norm_abs = torch.abs(gt_norm)
                            relative_mae = mae_loss / (gt_abs + 1e-6)  # Add small epsilon to avoid division by zero
                            relative_mae_norm = mae_loss_norm / (gt_norm_abs + 1e-6)
                            
                            # Create mask for non-zero ground truth values
                            non_zero_mask = (gt_abs > 1e-6)
                            non_zero_mask_norm = (gt_norm_abs > 1e-6)
                            
                            if non_zero_mask.any():
                                sample_relative_maes.append((relative_mae * non_zero_mask.float()).sum() / non_zero_mask.sum())
                            else:
                                sample_relative_maes.append(torch.tensor(0.0, device=self.accelerator.device))
                                
                            if non_zero_mask_norm.any():
                                sample_relative_maes_norm.append((relative_mae_norm * non_zero_mask_norm.float()).sum() / non_zero_mask_norm.sum())
                            else:
                                sample_relative_maes_norm.append(torch.tensor(0.0, device=self.accelerator.device))

                    if len(sample_losses) == 0:
                        # 没有可用 GT，跳过此 batch
                        self.accelerator.print(f"Warning: gt_ts_list has no valid entries for batch {i}, skipping...")
                        continue

                    batch_loss = torch.stack(sample_losses).mean()
                    batch_mae = torch.stack(sample_maes).mean()
                    batch_loss_norm = torch.stack(sample_losses_norm).mean()
                    batch_mae_norm = torch.stack(sample_maes_norm).mean()
                    batch_relative_mae = torch.stack(sample_relative_maes).mean()
                    batch_relative_mae_norm = torch.stack(sample_relative_maes_norm).mean()

                    per_batch_loss.append(batch_loss.detach().float().item())
                    per_batch_mae.append(batch_mae.detach().float().item())
                    per_batch_loss_norm.append(batch_loss_norm.detach().float().item())
                    per_batch_mae_norm.append(batch_mae_norm.detach().float().item())
                    per_batch_relative_mae.append(batch_relative_mae.detach().float().item())
                    per_batch_relative_mae_norm.append(batch_relative_mae_norm.detach().float().item())

                except Exception as e:
                    print(f"Full traceback:\n{traceback.format_exc()}")
                    self.save_error_batch(batch_data, None, epoch, i + 1, f"{phase}_error: {e}")
                    # For validation, we can continue with other batches instead of stopping
                    self.accelerator.print(f"Skipping batch {i + 1} and continuing {phase}...")
                    continue

        # Aggregate metrics across processes safely (use 1D tensors of scalars)
        device = self.accelerator.device
        if len(per_batch_loss) > 0:
            local_loss_tensor = torch.tensor(per_batch_loss, dtype=torch.float32, device=device)
            local_mae_tensor = torch.tensor(per_batch_mae, dtype=torch.float32, device=device)
            local_loss_norm_tensor = torch.tensor(per_batch_loss_norm, dtype=torch.float32, device=device)
            local_mae_norm_tensor = torch.tensor(per_batch_mae_norm, dtype=torch.float32, device=device)
            local_relative_mae_tensor = torch.tensor(per_batch_relative_mae, dtype=torch.float32, device=device)
            local_relative_mae_norm_tensor = torch.tensor(per_batch_relative_mae_norm, dtype=torch.float32, device=device)
        else:
            # Create empty tensors to allow gather to proceed
            local_loss_tensor = torch.tensor([], dtype=torch.float32, device=device)
            local_mae_tensor = torch.tensor([], dtype=torch.float32, device=device)
            local_loss_norm_tensor = torch.tensor([], dtype=torch.float32, device=device)
            local_mae_norm_tensor = torch.tensor([], dtype=torch.float32, device=device)
            local_relative_mae_tensor = torch.tensor([], dtype=torch.float32, device=device)
            local_relative_mae_norm_tensor = torch.tensor([], dtype=torch.float32, device=device)

        gathered_loss = self.accelerator.gather_for_metrics(local_loss_tensor)
        gathered_mae = self.accelerator.gather_for_metrics(local_mae_tensor)
        gathered_loss_norm = self.accelerator.gather_for_metrics(local_loss_norm_tensor)
        gathered_mae_norm = self.accelerator.gather_for_metrics(local_mae_norm_tensor)
        gathered_relative_mae = self.accelerator.gather_for_metrics(local_relative_mae_tensor)
        gathered_relative_mae_norm = self.accelerator.gather_for_metrics(local_relative_mae_norm_tensor)

        # Compute global averages on all processes consistently
        total_loss = gathered_loss.mean().item() if gathered_loss.numel() > 0 else float('inf')
        total_mae_loss = gathered_mae.mean().item() if gathered_mae.numel() > 0 else float('inf')
        total_loss_norm = gathered_loss_norm.mean().item() if gathered_loss_norm.numel() > 0 else float('inf')
        total_mae_loss_norm = gathered_mae_norm.mean().item() if gathered_mae_norm.numel() > 0 else float('inf')
        total_relative_mae = gathered_relative_mae.mean().item() if gathered_relative_mae.numel() > 0 else float('inf')
        total_relative_mae_norm = gathered_relative_mae_norm.mean().item() if gathered_relative_mae_norm.numel() > 0 else float('inf')

        del gathered_loss, gathered_mae, gathered_loss_norm, gathered_mae_norm, gathered_relative_mae, gathered_relative_mae_norm
        del local_loss_tensor, local_mae_tensor, local_loss_norm_tensor, local_mae_norm_tensor, local_relative_mae_tensor, local_relative_mae_norm_tensor

        self.model.train()
        return total_loss, total_mae_loss, total_loss_norm, total_mae_loss_norm, total_relative_mae, total_relative_mae_norm
    
    def save_validation_results_to_csv(self, epoch, test_results):
        """Save validation results to CSV file"""
        if not self.accelerator.is_main_process:
            return
            
        csv_filename = f'epoch_{epoch}_results.csv'
        csv_path = os.path.join(self.forecast_results_path, csv_filename)
        
        try:
            # Sort test results by dataset name (filename)
            sorted_test_results = sorted(test_results, key=lambda x: x['name'])
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['dataset_name', 'test_loss', 'test_mae_loss', 'test_loss_norm', 'test_mae_loss_norm', 'test_relative_mae', 'test_relative_mae_norm']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write results for each test dataset (sorted by name)
                for test_result in sorted_test_results:
                    writer.writerow({
                        'dataset_name': test_result['name'],
                        'test_loss': f"{test_result['test_loss']:.4f}",
                        'test_mae_loss': f"{test_result['test_mae_loss']:.4f}",
                        'test_loss_norm': f"{test_result['test_loss_norm']:.4f}",
                        'test_mae_loss_norm': f"{test_result['test_mae_loss_norm']:.4f}",
                        'test_relative_mae': f"{(test_result['test_relative_mae'] * 100):.2f}",
                        'test_relative_mae_norm': f"{(test_result['test_relative_mae_norm'] * 100):.2f}"
                    })
            
            self.accelerator.print(f"Validation results saved to: {csv_path}")
            
        except Exception as e:
            self.accelerator.print(f"Warning: Could not save validation results to CSV: {e}")
    
    def adjust_learning_rate_epoch(self, epoch):
        """Adjust learning rate for one epoch based on scheduler type"""
        if self.args.lradj not in ['TST', 'COSINE_WARMUP']:
            if self.args.lradj == 'COS':
                self.scheduler.step()
                self.accelerator.print(f"lr = {self.model_optim.param_groups[0]['lr']:.10f}")
            else:
                if epoch == 0:
                    self.args.learning_rate = self.model_optim.param_groups[0]['lr']
                    self.accelerator.print(f"lr = {self.model_optim.param_groups[0]['lr']:.10f}")
                adjust_learning_rate(self.accelerator, self.model_optim, self.scheduler, epoch + 1, self.args, printout=True)
        elif self.args.lradj == 'COSINE_WARMUP':
            self.accelerator.print(f"lr = {self.model_optim.param_groups[0]['lr']:.10f}")
        else:
            self.accelerator.print(f'Updating learning rate to {self.scheduler.get_last_lr()[0]}')
    
    def run(self):
        """Run a single experiment"""
        try:
            # Training loop
            for epoch in range(self.args.start_epoch, self.args.train_epochs):
                train_loss, predict_loss_list, analysis_loss_list = self.train_epoch(epoch)
                
                try:
                    self.save_checkpoint(epoch + 1)
                except Exception as e:
                    print(f"Full traceback:\n{traceback.format_exc()}")

                # Compute average losses
                train_loss = np.average(train_loss)
                
                if self.args.analysis_only:
                    analysis_loss_avg = np.average(analysis_loss_list)
                    self.accelerator.print(f"Analysis Loss: {analysis_loss_avg:.7f}")
                elif self.args.add_analysis:
                    predict_loss_avg = np.average(predict_loss_list)
                    analysis_loss_avg = np.average(analysis_loss_list)
                    self.accelerator.print(f"Predict Loss: {predict_loss_avg:.7f} | Analysis Loss: {analysis_loss_avg:.7f}")
                
                # Validation (skip for analysis_only mode as it requires forecast data)
                if self.args.analysis_only:
                    vali_loss, vali_mae_loss, test_results, vali_loss_norm, vali_mae_loss_norm = 0.0, 0.0, [], 0.0, 0.0
                    self.accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} (Analysis-only mode)")
                else:
                    vali_loss, vali_mae_loss, test_results, vali_loss_norm, vali_mae_loss_norm = self.validate_epoch(epoch + 1)
                    
                    # Save validation results to CSV file instead of printing detailed results
                    if test_results:
                        self.save_validation_results_to_csv(epoch + 1, test_results)
                        # Print summary information in log
                        self.accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Vali MAE Loss: {vali_mae_loss:.7f}")
                        self.accelerator.print(f"Epoch: {epoch + 1} | Validated {len(test_results)} test datasets, results saved to CSV")
                    else:
                        self.accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Vali MAE Loss: {vali_mae_loss:.7f}")
                        self.accelerator.print(f"Epoch: {epoch + 1} | No test datasets to validate")
                
                # Early stopping (if enabled and not analysis_only)
                if self.early_stopping is not None and not self.args.analysis_only:
                    self.early_stopping(vali_loss)
                    if self.early_stopping.early_stop:
                        self.accelerator.print("Early stopping")
                        break
                
                # Adjust learning rate
                self.adjust_learning_rate_epoch(epoch)
            
            # Save training summary
            summary_kwargs = {}
            if self.args.analysis_only:
                summary_kwargs['analysis_loss_avg'] = analysis_loss_avg if 'analysis_loss_avg' in locals() else None
            elif self.args.add_analysis:
                summary_kwargs['predict_loss_avg'] = predict_loss_avg if 'predict_loss_avg' in locals() else None
                summary_kwargs['analysis_loss_avg'] = analysis_loss_avg if 'analysis_loss_avg' in locals() else None
            
            # Use first test result for summary, or default values if no test results
            if test_results:
                first_test_result = test_results[0]
                summary_test_loss = first_test_result['test_loss']
                summary_test_mae_loss = first_test_result['test_mae_loss']
            else:
                summary_test_loss = 0.0
                summary_test_mae_loss = 0.0
            
            self.logger.save_training_summary(
                self.setting, epoch, train_loss, vali_loss, summary_test_loss, 
                summary_test_mae_loss, self.early_stopping, self.args, **summary_kwargs
            )
            
        except Exception as e:
            print(f"Full traceback:\n{traceback.format_exc()}")
        
        finally:
            # Always restore stdout and cleanup
            self.logger.restore_stdout()
            self.accelerator.wait_for_everyone()
        

def run_all_experiments(args):
    """Run all experiments with different iterations"""
    for ii in range(args.itr):
        experiment = TimeOmniExperiment(args, ii)
        experiment.run()


def valid_experiment(args):
    exp = TimeOmniExperiment(args)
    _, _, test_results, _, _ = exp.validate_epoch()
    
    # Save results to CSV even in test-only mode
    if test_results:
        exp.save_validation_results_to_csv(args.start_epoch, test_results)
        print(f"Test results saved to CSV in: {exp.forecast_results_path}")


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description='TimeOmni')
    
    # Set random seed
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # Basic config
    parser.add_argument('--model', type=str, required=True, default='TimeOmni', help='model name, options: [TimeOmni]')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--timestamp', type=str, default=None, help='experiment timestamp for consistent folder naming across processes')
    parser.add_argument('--exp_dir', type=str, default='./exp/', help='location of exp dir')
    
    # Model define
    parser.add_argument('--pred_len', type=int, nargs='+', default=[720], help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_model', type=str, default='qwen3', help='LLM model')
    # Analysis patch configs now support multiple sizes; pass as space-separated ints, e.g. --analysis_patch_len 640 1024 --analysis_stride 320 512
    parser.add_argument('--patch_nums', type=int, default=50, help='number of patches for time series input')
    parser.add_argument('--ts_tokens', type=int, default=100, help='number of tokens for time series input')
    parser.add_argument('--analysis_patch_len', type=int, nargs='+', default=[640], help='one or more patch lengths for analysis task')
    parser.add_argument('--analysis_stride', type=int, nargs='+', default=[320], help='one or more strides for analysis task (aligned with analysis_patch_len)')
    parser.add_argument('--analysis_d_model', type=int, default=512, help='fixed d_model for analysis PatchEmbedding/ReprogrammingLayer')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--analysis_batch_size', type=int, default=6, help='analysis task batch size of train input data')
    parser.add_argument('--forecast_batch_size', type=int, default=1, help='forecast task batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--use_early_stop', action='store_true', help='use early stopping', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='COSINE_WARMUP', help='adjust learning rate')
    parser.add_argument('--max_keep_epochs', type=int, default=4, help='maximum number of epoch checkpoints to keep')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--warmup_steps', type=int, help='number of warmup steps for cosine schedule with warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='warmup ratio for cosine schedule with warmup')
    parser.add_argument('--analysis_loss_weight', type=float, default=1.0, help='weight for analysis loss in combined loss')
    
    # DoRA fine-tuning parameters
    parser.add_argument('--use_dora', action='store_true', help='use DoRA for LLM fine-tuning', default=False)
    parser.add_argument('--dora_r', type=int, default=16, help='rank for DoRA adaptation')
    parser.add_argument('--dora_alpha', type=int, default=32, help='alpha for DoRA scaling')
    parser.add_argument('--dora_dropout', type=float, default=0.1, help='dropout for DoRA layers')
    parser.add_argument('--dora_target_modules', type=str, default='q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj', help='target modules for DoRA')
    parser.add_argument('--full_finetune', action='store_true', help='fine-tune all LLM parameters instead of using DoRA', default=False)
    
    # Ablation parameters
    parser.add_argument('--use_mlp_reprogramming', action='store_true', help='use MLP-based ReprogrammingLayer instead of attention-based (for ablation study)', default=False)
    
    # Checkpoint resuming parameters
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to checkpoint file to resume training from, format: exp/setting/timestamp/epoch_x')
    parser.add_argument('--load_training_states', action='store_true', help='load optimizer, scheduler, scaler states from checkpoint (use with --ckpt_path)', default=False)
    parser.add_argument('--start_epoch', default=0, help='epoch to start training from (use with --load_training_states), start_epoch is x when ckpt is epoch_x')
    parser.add_argument('--test_only', action='store_true', help='only test the model, no training', default=False)
    parser.add_argument('--forecast_results_dir', type=str, default='forecast_results', help='directory name to save forecasting results under exp/setting/')
    
    # dataset parameters
    parser.add_argument('--add_analysis', action='store_true', help='add analysis task with dual dataloader training', default=False)
    parser.add_argument('--analysis_only', action='store_true', help='train only analysis task (no forecasting)', default=False)
    parser.add_argument('--forecast_jsonl_file_path', type=str, help='path to JSONL file for forecasting train dataset')
    parser.add_argument('--forecast_val_jsonl_file_path', type=str, help='path to JSONL file for forecasting valid dataset')
    parser.add_argument('--forecast_test_jsonl_file_path', type=str, nargs='+', help='path(s) to JSONL file(s) for forecasting test dataset(s)')
    parser.add_argument('--analysis_jsonl_file_path', type=str, help='path to JSONL file for analysis dataset')
    parser.add_argument('--base_dir', type=str, help='directory containing ts files')
    parser.add_argument('--unfold_channels', action='store_true', help='unfold input time series channels dim to time dim, if true, from (T, C) to (T * C, 1)', default=True)
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.test_only:
        valid_experiment(args)
    else:
        run_all_experiments(args)
