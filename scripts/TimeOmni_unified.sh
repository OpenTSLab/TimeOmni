cd ~/TimeOmni
source ~/miniconda3/bin/activate timeomni
export TOKENIZERS_PARALLELISM=false

train_epochs=10
max_keep_epochs=5
learning_rate=2e-5
batch_size=7
analysis_batch_size=6
forecast_batch_size=1
eval_batch_size=1 # must be 1 for variable-length inputs
exp_dir=exp
lradj="COSINE_WARMUP"
model_name=TimeOmni
llm_model=qwen3
comment='benchmark-13forecast_v3-21analysis_v2-dora_rank8-ts_tokens100-wo_router-patchlen1024'

start_epoch=10
ckpt_path=exp/benchmark-8forecast-22analysis/lr2e-05_COSINE_WARMUP_bs32_epochs10/exp0/epoch_10

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
  --analysis_jsonl_file_path dataset/merged_train_analysis_v2.jsonl \
  --forecast_jsonl_file_path dataset/merged_train_forecast_v3.jsonl \
  --forecast_val_jsonl_file_path dataset/Release_v1/textETT-Energy-Reverse_Forecasting-textETT/textETT-Energy-Reverse_Forecasting-textETT-test-2700.jsonl \
  --forecast_test_jsonl_file_path \
  dataset/Release_v1/Chaotic-Math-Forecasting-Chaotic_system/Chaotic-Math-Forecasting-Chaotic_system-test-1856.jsonl \
  dataset/Release_v1/NewsForecast-Energy-Forecasting-Electronic_load/NewsForecast-Energy-Forecasting-Electronic_load-test-100.jsonl \
  dataset/Release_v1/TS_MQA-Neuroscience-Forecasting-EEG/TS_MQA-Neuroscience-Forecasting-EEG-test-360.jsonl \
  dataset/Release_v1/TS_MQA-Neuroscience-Imputation-EEG/TS_MQA-Neuroscience-Imputation-EEG-test-373.jsonl \
  dataset/Release_v1/TS_MQA-Physiology-Forecasting-Health/TS_MQA-Physiology-Forecasting-Health-test-1517.jsonl \
  dataset/Release_v1/TS_MQA-Physiology-Imputation-Health/TS_MQA-Physiology-Imputation-Health-test-1511.jsonl \
  dataset/Release_v1/textETT-Energy-Reverse_Forecasting-textETT/textETT-Energy-Reverse_Forecasting-textETT-test-2700.jsonl \
  dataset/Release_v1/ETT-Energy-Forecasting-ETT/ETT-Energy-Forecasting-ETT-test-5570.jsonl \
  dataset/Release_v1/MetroTraffic-Urbanism-Forecasting-Traffic/MetroTraffic-Urbanism-Forecasting-Traffic-test-3200.jsonl \
  dataset/Release_v1/TS_MQA-Energy-Forecasting-Electricity/TS_MQA-Energy-Forecasting-Electricity-test-66.jsonl \
  dataset/Release_v1/TS_MQA-Energy-Imputation-Electricity/TS_MQA-Energy-Imputation-Electricity-test-65.jsonl \
  dataset/Release_v1/TS_MQA-Urbanism-Forecasting-Pedestrian_activity/TS_MQA-Urbanism-Forecasting-Pedestrian_activity-test-62.jsonl \
  dataset/Release_v1/TS_MQA-Urbanism-Imputation-Pedestrian_activity/TS_MQA-Urbanism-Imputation-Pedestrian_activity-test-65.jsonl \
  dataset/Release_v1/FinMultiTime-Economics-Forecasting-Stock_closing_price/FinMultiTime-Economics-Forecasting-Stock_closing_price-test-1260.jsonl \
  dataset/Release_v1/MT_bench-Economics-Forecasting-Stock_price/MT_bench-Economics-Forecasting-Stock_price-test-383.jsonl \
  dataset/Release_v1/MT_bench-Meteorology-Forecasting-Temperature/MT_bench-Meteorology-Forecasting-Temperature-test-1176.jsonl \
  dataset/Release_v1/NewsForecast-Urbanism-Forecasting-Traffic_flow/NewsForecast-Urbanism-Forecasting-Traffic_flow-test-43.jsonl \
  --base_dir dataset/Release_v1 \
  --exp_dir $exp_dir \
  --lradj $lradj \
  --timestamp $timestamp \
  --unfold_channels \
  --analysis_stride 1024 \
  --analysis_patch_len 1024 \
  --pred_len 1 2 4 8 16 32 64 128 256 512 1024 \
  --patch_nums 50 \
  --ts_tokens 100 \
  --add_analysis \
  --use_dora \
