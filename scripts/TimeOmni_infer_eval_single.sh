#!/bin/bash

cd ~/TimeOmni
source ~/miniconda3/bin/activate timeomni
export TOKENIZERS_PARALLELISM=false

infer=true
eval=true

model_dirs="exp/benchmark-analysis-RadarCom/lr2e-05_COSINE_WARMUP_bs8_epochs10
"

for model_dir in $model_dirs; do
    echo "Processing model directory: $model_dir"
    for epoch in 10; do
        echo "Evaluating epoch: $epoch"
        model_path="$model_dir/exp0/epoch_$epoch/pytorch_model/mp_rank_00_model_states.pt"
        infer_folder="$model_dir/exp0/infer_results/epoch_$epoch"
        eval_output="$model_dir/exp0/eval_results/epoch_$epoch-eval_results.csv"

        if [ "$infer" = true ]; then
            torchrun \
            infer_benchmark.py \
            --data_file_list \
            dataset/Release_v1/RadarCom-Radar-Classification-Radar_signal/RadarCom-Radar-Classification-Radar_signal-test-1280.jsonl \
            --media_root dataset/Release_v1 \
            --output_folder $infer_folder \
            --model_path $model_path \
            --batch_size 128 \
            --max_new_tokens 50 \
            --num_workers 16
            echo "Inference completed! Results saved to $infer_folder"
        fi

        if [ "$eval" = true ]; then
            python eval_benchmark.py "$infer_folder" --output "$eval_output"
            echo "Evaluation completed! Results saved to $eval_output"
        fi
    done
done