from torch.utils.data import DataLoader

from .dataset import Dataset_Unified, Collator_Unified


def data_provider(args, flag, tokenizer=None):
    """
    统一的数据提供器，根据flag和参数配置返回相应的数据集和数据加载器
    
    训练数据分为三种情况：
    1. 只有分析jsonl（analysis_only=True）
    2. 只有forecast jsonl（默认情况）
    3. 两者都有（add_analysis=True）
    
    验证和测试数据只有forecast jsonl，或者全部为None
    """

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size

    # 根据flag确定数据文件路径
    if flag == 'train':
        # 训练阶段：根据配置决定使用哪些数据集
        
        # 情况1：只有分析数据
        if getattr(args, 'analysis_only', False):
            if not hasattr(args, 'analysis_jsonl_file_path'):
                raise ValueError("analysis_only=True but analysis_jsonl_file_path not provided")
            
            dataset_analysis = Dataset_Unified(
                jsonl_file_path=args.analysis_jsonl_file_path,
                tokenizer=tokenizer,
                base_dir=getattr(args, 'base_dir', None),
                use_scaler=True,
                unfold_channels=getattr(args, 'unfold_channels', True),
                # patch_len=getattr(args, 'analysis_patch_len', None)
            )
            
            collator = Collator_Unified(tokenizer=tokenizer)
            
            analysis_loader = DataLoader(
                dataset_analysis,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=getattr(args, 'num_workers', 0),
                collate_fn=collator,
                drop_last=drop_last
            )
            
            return None, None, dataset_analysis, analysis_loader
        
        # 情况2和3：有forecast数据（可能还有分析数据）
        else:
            if not hasattr(args, 'forecast_jsonl_file_path'):
                raise ValueError("forecast_jsonl_file_path not provided for training")
            
            dataset_forecast = Dataset_Unified(
                jsonl_file_path=args.forecast_jsonl_file_path,
                tokenizer=tokenizer,
                base_dir=getattr(args, 'base_dir', None),
                use_scaler=True,
                unfold_channels=False,
                # patch_len=getattr(args, 'patch_len', None)
            )
            
            collator_forecast = Collator_Unified(tokenizer=tokenizer, pad_side_time='left')
            
            # 情况3：同时有分析数据
            if getattr(args, 'add_analysis', False):
                if not hasattr(args, 'analysis_jsonl_file_path'):
                    raise ValueError("add_analysis=True but analysis_jsonl_file_path not provided")
                
                dataset_analysis = Dataset_Unified(
                    jsonl_file_path=args.analysis_jsonl_file_path,
                    tokenizer=tokenizer,
                    base_dir=getattr(args, 'base_dir', None),
                    use_scaler=True,
                    unfold_channels=getattr(args, 'unfold_channels', True),
                    # patch_len=getattr(args, 'analysis_patch_len', None)
                )

                collator_analysis = Collator_Unified(tokenizer=tokenizer, pad_side_time='right')
                
                # # 计算各数据集的batch size比例
                # forecast_dataset_len = len(dataset_forecast)
                # analysis_dataset_len = len(dataset_analysis)
                # total_dataset_len = forecast_dataset_len + analysis_dataset_len
                
                # forecast_batch_size = round(batch_size * forecast_dataset_len / total_dataset_len)
                # analysis_batch_size = round(batch_size * analysis_dataset_len / total_dataset_len)
                
                # # 确保batch size至少为1
                # if forecast_batch_size == 0:
                #     forecast_batch_size = 1
                # if analysis_batch_size == 0:
                #     analysis_batch_size = 1
                
                forecast_loader = DataLoader(
                    dataset_forecast,
                    batch_size=args.forecast_batch_size,
                    shuffle=shuffle_flag,
                    num_workers=getattr(args, 'num_workers', 0),
                    collate_fn=collator_forecast,
                    drop_last=drop_last
                )
                
                analysis_loader = DataLoader(
                    dataset_analysis,
                    batch_size=args.analysis_batch_size,
                    shuffle=shuffle_flag,
                    num_workers=getattr(args, 'num_workers', 0),
                    collate_fn=collator_analysis,
                    drop_last=drop_last
                )
                
                return dataset_forecast, forecast_loader, dataset_analysis, analysis_loader
            
            # 情况2：只有forecast数据
            else:
                forecast_loader = DataLoader(
                    dataset_forecast,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=getattr(args, 'num_workers', 0),
                    collate_fn=collator_forecast,
                    drop_last=drop_last
                )
                
                return dataset_forecast, forecast_loader, None, None
    
    # 验证和测试阶段：只有forecast数据
    else:
        # 如果是只有分析数据的情况，直接返回None
        if getattr(args, 'analysis_only', False):
            return None, None, None, None
        
        if flag == 'val':
            if not hasattr(args, 'forecast_val_jsonl_file_path'):
                raise ValueError("forecast_val_jsonl_file_path not provided for validation")
            jsonl_file_path = args.forecast_val_jsonl_file_path
        elif flag == 'test':
            if not hasattr(args, 'forecast_test_jsonl_file_path'):
                raise ValueError("forecast_test_jsonl_file_path not provided for testing")
            jsonl_file_path = args.forecast_test_jsonl_file_path
        else:
            raise ValueError(f"Unknown flag: {flag}")
        
        dataset_forecast = Dataset_Unified(
            jsonl_file_path=jsonl_file_path,
            tokenizer=tokenizer,
            base_dir=getattr(args, 'base_dir', None),
            use_scaler=False,
            unfold_channels=False,
            max_tokens=100000 # if larger than 100000, will OOM(140GB RAM)
            # patch_len=getattr(args, 'patch_len', None)
        )

        # dataset_forecast_unnorm = Dataset_Unified(
        #     jsonl_file_path=jsonl_file_path,
        #     tokenizer=tokenizer,
        #     base_dir=getattr(args, 'base_dir', None),
        #     use_scaler=False,
        #     unfold_channels=getattr(args, 'unfold_channels', True),
        #     # patch_len=getattr(args, 'patch_len', None)
        # )

        collator = Collator_Unified(tokenizer=tokenizer, pad_side_time='left')

        forecast_loader = DataLoader(
            dataset_forecast,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=getattr(args, 'num_workers', 0),
            collate_fn=collator,
            drop_last=drop_last
        )

        # forecast_loader_unnorm = DataLoader(
        #     dataset_forecast_unnorm,
        #     batch_size=batch_size,
        #     shuffle=shuffle_flag,
        #     num_workers=getattr(args, 'num_workers', 0),
        #     collate_fn=collator,
        #     drop_last=drop_last
        # )

        # return dataset_forecast, forecast_loader, dataset_forecast_unnorm, forecast_loader_unnorm
        return dataset_forecast, forecast_loader, None, None