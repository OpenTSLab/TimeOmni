from torch.utils.data import DataLoader

from .dataset import Dataset_Unified, Collator_Unified


def data_provider(args, flag, tokenizer=None):
    """
    Unified data provider that returns datasets and loaders based on flag and args.

    Training data cases:
    1. Analysis-only JSONL (analysis_only=True)
    2. Forecast-only JSONL (default)
    3. Both analysis and forecast (add_analysis=True)

    Validation and test data use forecast JSONL only, or return None.
    """

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size

    # Determine data file paths based on flag.
    if flag == 'train':
        # Training phase: choose datasets based on configuration.
        
        # Case 1: analysis-only data.
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
        
        # Cases 2 and 3: forecast data (optionally with analysis data).
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
            
            # Case 3: both analysis and forecast data.
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
            
            # Case 2: forecast-only data.
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
    
    # Validation and test: forecast data only.
    else:
        # If analysis-only, return None.
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
        )

        collator = Collator_Unified(tokenizer=tokenizer, pad_side_time='left')

        forecast_loader = DataLoader(
            dataset_forecast,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=getattr(args, 'num_workers', 0),
            collate_fn=collator,
            drop_last=drop_last
        )

        return dataset_forecast, forecast_loader, None, None