#!/bin/bash

# 时序数据推理评估脚本

cd ~/TimeOmni
source ~/miniconda3/bin/activate timeomni
export TOKENIZERS_PARALLELISM=false

# ---------- Dynamic master port selection (avoid conflicts) ----------
# Honor externally provided MASTER_PORT or MAIN_PROCESS_PORT if set.
SHARED_DIR="~/TimeOmni/exp"
PORT_FILE="$SHARED_DIR/.master_port_current_run"
DEFAULT_PORT=29500

# Normalize any provided alias
[ -n "$MAIN_PROCESS_PORT" ] && export MASTER_PORT="$MAIN_PROCESS_PORT"

if [ -z "$MASTER_PORT" ]; then
  if [ "${NODE_RANK:-0}" = "0" ]; then
    # Node 0 chooses a free port
    CHOSEN_PORT=$(python - <<'EOF'
import socket, random
candidates = list(range(20000,40000))
random.shuffle(candidates)
for p in candidates[:200]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('', p))
        except OSError:
            continue
        print(p)
        break
EOF
)
    if [ -z "$CHOSEN_PORT" ]; then
      CHOSEN_PORT=$DEFAULT_PORT
    fi
    echo -n "$CHOSEN_PORT" > "$PORT_FILE"
  else
    # Other nodes wait for port file
    for i in $(seq 1 60); do
      [ -f "$PORT_FILE" ] && break
      sleep 1
    done
    CHOSEN_PORT=$(cat "$PORT_FILE" 2>/dev/null || echo "$DEFAULT_PORT")
  fi
  export MASTER_PORT="$CHOSEN_PORT"
fi

# Safety echo (optional - uncomment if debugging)
# echo "Using MASTER_PORT=$MASTER_PORT on NODE_RANK=${NODE_RANK:-0}"

# ---------- End dynamic port selection ----------

# Multi-node distributed training configuration
# Use platform-injected environment variables
WORLD_SIZE=$((NODE_COUNT * PROC_PER_NODE))

infer=true
eval=true

model_dirs="exp/benchmark-13forecast_v3-21analysis_v2-dora_rank8-ts_tokens100-wo_router-patchlen1024/lr2e-05_COSINE_WARMUP_bs7_epochs10
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
            --nnodes=$NODE_COUNT \
            --node_rank=$NODE_RANK \
            --nproc_per_node=$PROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            infer_benchmark.py \
            --data_file_list \
            dataset/Release_v1/CWRU-Manufacturing-Classification-Industrial_bearings/CWRU-Manufacturing-Classification-Industrial_bearings-test-400.jsonl \
            dataset/Release_v1/GWOSC_GW_Event-Astronomy-Event_Detection-Gravitational_Wave/GWOSC_GW_Event-Astronomy-Event_Detection-Gravitational_Wave-test-1024.jsonl \
            dataset/Release_v1/iNaturalist-Bioacoustics-Classification-Animal_sound/iNaturalist-Bioacoustics-Classification-Animal_sound-test-2474.jsonl \
            dataset/Release_v1/LEAVES-Astronomy-Classification-Light_Curve/LEAVES-Astronomy-Classification-Light_Curve-test-1024.jsonl \
            dataset/Release_v1/MarmAudio-Bioacoustics-Classification-Marmoset/MarmAudio-Bioacoustics-Classification-Marmoset-test-900.jsonl \
            dataset/Release_v1/MDD-Neuroscience-Anomaly_Detection-Depressive_disorder/MDD-Neuroscience-Anomaly_Detection-Depressive_disorder-test-1000.jsonl \
            dataset/Release_v1/MIMII_Due-Manufacturing-Anomaly_Detection-Industrial_machine/MIMII_Due-Manufacturing-Anomaly_Detection-Industrial_machine-test-3600.jsonl \
            dataset/Release_v1/MT_bench-Economics-QA-Stock/MT_bench-Economics-QA-Stock-test-303.jsonl \
            dataset/Release_v1/MT_bench-Meteorology-QA-Temperature/MT_bench-Meteorology-QA-Temperature-test-413.jsonl \
            dataset/Release_v1/Powdermill-Bioacoustics-Classification-Birds_sound/Powdermill-Bioacoustics-Classification-Birds_sound-test-1602.jsonl \
            dataset/Release_v1/PTB_XL-Physiology-Classification-ECG/PTB_XL-Physiology-Classification-ECG-test-2090.jsonl \
            dataset/Release_v1/RadarCom-Radar-Classification-Radar_signal/RadarCom-Radar-Classification-Radar_signal-test-1280.jsonl \
            dataset/Release_v1/RadSeg-Radar-Classification-Radar_segment/RadSeg-Radar-Classification-Radar_segment-test-963.jsonl \
            dataset/Release_v1/Sleep-Neuroscience-Classification-Sleep_stage/Sleep-Neuroscience-Classification-Sleep_stage-test-1024.jsonl \
            dataset/Release_v1/STEAD-Earth_Science-Event_Detection-Earthquake/STEAD-Earth_Science-Event_Detection-Earthquake-test-2048.jsonl \
            dataset/Release_v1/TIMECAP-Meteorology-Anomaly_Detection-Rainfall/TIMECAP-Meteorology-Anomaly_Detection-Rainfall-test-1131.jsonl \
            dataset/Release_v1/TS_MQA-Meteorology-Anomaly_Detection-Weather/TS_MQA-Meteorology-Anomaly_Detection-Weather-test-1256.jsonl \
            dataset/Release_v1/TS_MQA-Physiology-Anomaly_Detection-ECG/TS_MQA-Physiology-Anomaly_Detection-ECG-test-1006.jsonl \
            dataset/Release_v1/TS_MQA-Physiology-Anomaly_Detection-Freezing/TS_MQA-Physiology-Anomaly_Detection-Freezing-test-1882.jsonl \
            dataset/Release_v1/TS_MQA-Physiology-Classification-Activity/TS_MQA-Physiology-Classification-Activity-test-1818.jsonl \
            dataset/Release_v1/TS_MQA-Urbanism-Anomaly_Detection-Traffic_flow/TS_MQA-Urbanism-Anomaly_Detection-Traffic_flow-test-122.jsonl \
            dataset/Release_v1/TUEV-Neuroscience-Classification-EEG_waveform/TUEV-Neuroscience-Classification-EEG_waveform-test-789.jsonl \
            dataset/Release_v1/WBCIC_SHU-Neuroscience-Classification-Movement_imagination/WBCIC_SHU-Neuroscience-Classification-Movement_imagination-test-3600.jsonl \
            --media_root dataset/Release_v1 \
            --output_folder $infer_folder \
            --model_path $model_path \
            --batch_size 128 \
            --max_new_tokens 200 \
            --num_workers 16
            echo "Inference completed! Results saved to $infer_folder"
        fi

        if [ "$eval" = true ]; then
            python eval_benchmark.py "$infer_folder" --output "$eval_output"
            echo "Evaluation completed! Results saved to $eval_output"
        fi
    done
done