import json
import os
from tqdm import tqdm
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from models import TimeOmni
from data_provider.dataset import Dataset_Unified, Collator_Unified


def setup_distributed():
    """设置分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    return True, rank, world_size, gpu


def create_dataloader(jsonl_file, tokenizer, media_root, batch_size, num_workers=16, distributed=False):
    """为单个jsonl创建DataLoader"""
    dataset = Dataset_Unified(
        tokenizer=tokenizer,
        jsonl_file_path=jsonl_file,
        base_dir=media_root,
        use_scaler=True,
        unfold_channels=True,
    )
    sampler = DistributedSampler(dataset) if distributed else None
    data_collator = Collator_Unified(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator
    )
    
    return dataloader


def inference_single_jsonl(model, dataloader, jsonl_file, generation_config, rank=0):
    """对单个jsonl进行推理"""
    results = []
    
    if rank == 0:
        print(f"\nProcessing jsonl file: {jsonl_file}")
        print(f"Number of batches: {len(dataloader)}")

    for batch in tqdm(dataloader, desc=f"Processing {jsonl_file}", disable=(rank != 0)):
        try:
            input_ts = batch['input_ts_list']
            input_ts = [input.cuda() for input in input_ts]

            # 批处理推理 - 使用input_texts作为问题
            with torch.no_grad():
                responses = model.generate(
                    input_ts,
                    batch['input_texts'],
                    generation_config=generation_config
                )
            
            # 保存结果
            for i in range(len(batch['ids'])):
                results.append({
                    'id': batch['ids'][i],
                    "uid": batch['uids'][i],
                    "dataset_name": batch['dataset_names'][i],
                    "task": batch['tasks'][i],
                    "scene": batch['scenes'][i],
                    'ts_path': batch['input_ts_paths'][i],
                    'input_text': batch['input_texts'][i],
                    'generated_text': responses[i].strip(),
                    'ground_truth': batch['gt_texts'][i],
                    'gt_result': batch['gt_results'][i],
                })
        except Exception as e:
            print(f"[ERROR] Exception in batch inference for '{jsonl_file}': {e}")
            print(f"[ERROR] Batch IDs: {batch['ids']}")
            # 错误时不保存当前batch的任何结果，直接跳过
    return results


def save_results(results, output_file):
    """保存结果到jsonl文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Time Series LLM Evaluation with DDP")
    parser.add_argument("--model_path", type=str,
                       help="Path to the model")
    parser.add_argument("--data_file_list", type=str, nargs='+', required=True,
                       help="List of JSONL data files for inference")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Folder to save the results")
    parser.add_argument("--media_root", type=str, default=None,
                       help="Root path to prepend to relative ts_path in data")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Max new tokens for generation")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of workers for DataLoader")

    args = parser.parse_args()

    # 设置分布式环境
    distributed, rank, world_size, gpu = setup_distributed()

    # Load model
    if rank == 0:
        print("Loading model...")

    model = TimeOmni.Model.load_checkpoint(
        args.model_path,
    ).eval().to('cuda')

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    if rank == 0:
        print("Model loaded successfully!")

    # 逐个处理 data_file_list
    for data_file in args.data_file_list:
        if rank == 0:
            print(f"Loading dataset from {data_file}...")
            if args.media_root:
                print(f"Using media root: {args.media_root}")

        # Run inference
        if rank == 0:
            print(f"Starting inference for {data_file}...")

        dataloader = create_dataloader(
            jsonl_file=data_file,
            tokenizer=model.tokenizer,
            media_root=args.media_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed
        )

        results = inference_single_jsonl(
            model=model,
            dataloader=dataloader,
            jsonl_file=data_file,
            generation_config=generation_config,
            rank=rank
        )

        # 收集所有进程的结果
        if distributed:
            dist.barrier()
            all_results = [None] * world_size
            dist.all_gather_object(all_results, results)
            if rank == 0:
                results = []
                for rank_results in all_results:
                    results.extend(rank_results)

        # 只有主进程保存结果
        if rank == 0:
            # 生成输出文件名
            base_name = os.path.basename(data_file)
            output_file = os.path.join(args.output_folder, base_name)
            save_results(results, output_file)

if __name__ == "__main__":
    main()
