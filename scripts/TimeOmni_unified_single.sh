cd ~/TimeOmni
source ~/miniconda3/bin/activate timeomni
export TOKENIZERS_PARALLELISM=false

train_epochs=10
max_keep_epochs=1
learning_rate=2e-5
batch_size=4
analysis_batch_size=6
forecast_batch_size=1
eval_batch_size=1 # must be 1 for variable-length inputs
exp_dir=exp
lradj="COSINE_WARMUP"
model_name=TimeOmni
llm_model=qwen3
comment='benchmark-forecast-textETT-fixed-10'

timestamp=$(date +"%Y%m%d_%H%M%S")

# Multi-node distributed training configuration
# Use platform-injected environment variables
WORLD_SIZE=$((NODE_COUNT * PROC_PER_NODE))

accelerate launch \
  --mixed_precision bf16 \
  --num_processes $WORLD_SIZE \
  --num_machines $NODE_COUNT \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port 29500 \
  run_main_refactored_unified.py \
  --model $model_name \
  --llm_model $llm_model \
  --itr 1 \
  --d_ff 128 \
  --batch_size $batch_size \
  --forecast_batch_size $forecast_batch_size \
  --analysis_batch_size $analysis_batch_size \
  --eval_batch_size $eval_batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --max_keep_epochs $max_keep_epochs \
  --dora_r 8 \
  --dora_alpha 32 \
  --dora_dropout 0.1 \
  --model_comment $comment \
  --analysis_jsonl_file_path dataset/Release_train_standard/TUEV-Neuroscience-Classification-EEG_waveform-train-1592.jsonl \
  --forecast_jsonl_file_path dataset/Release_train_standard/textETT-Energy-Reverse_Forecasting-textETT-train-258010.jsonl \
  --forecast_val_jsonl_file_path dataset/Release_v1/textETT-Energy-Reverse_Forecasting-textETT/textETT-Energy-Reverse_Forecasting-textETT-test-2700.jsonl \
  --forecast_test_jsonl_file_path \
  dataset/Release_v1/textETT-Energy-Reverse_Forecasting-textETT/textETT-Energy-Reverse_Forecasting-textETT-test-2700.jsonl \
  --base_dir dataset/Release_v1 \
  --exp_dir $exp_dir \
  --lradj $lradj \
  --timestamp $timestamp \
  --unfold_channels \
  --analysis_stride 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --analysis_patch_len 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --pred_len 1 2 4 8 16 32 64 128 256 512 1024 \
  --patch_nums 50 \
  --ts_tokens 100 \
  --use_dora \
