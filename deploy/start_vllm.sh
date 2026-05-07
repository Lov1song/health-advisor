#!/usr/bin/env bash
# Phase 4: 直接启动 vLLM (不使用 Docker)
#
# 前提:
#   pip install vllm
#   已完成模型合并 (models/mental_health_merged/)
#
# 用法:
#   bash deploy/start_vllm.sh
#   bash deploy/start_vllm.sh --model /data/models/mental_health_merged --port 8001
#
# 环境变量覆盖:
#   MODEL_PATH=/path/to/model bash deploy/start_vllm.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$(pwd)/models/mental_health_merged}"
SERVED_NAME="${SERVED_NAME:-deepseek-8b-qlora}"
PORT="${PORT:-8001}"
GPU_MEM="${GPU_MEM:-0.88}"
MAX_LEN="${MAX_LEN:-2048}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --port)  PORT="$2";       shift 2 ;;
    --gpu-mem) GPU_MEM="$2";  shift 2 ;;
    *) shift ;;
  esac
done

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[错误] 模型目录不存在: $MODEL_PATH"
  echo "请先运行: python train/merge_lora.py && llamafactory-cli export train/export_config.yaml"
  exit 1
fi

echo "=============================="
echo " vLLM 启动参数"
echo "=============================="
echo "  模型路径  : $MODEL_PATH"
echo "  服务名称  : $SERVED_NAME"
echo "  端口      : $PORT"
echo "  GPU 内存  : $GPU_MEM"
echo "  最大长度  : $MAX_LEN"
echo ""
echo "API 文档: http://localhost:$PORT/docs"
echo "健康检查: http://localhost:$PORT/health"
echo "=============================="
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model              "$MODEL_PATH"   \
  --served-model-name  "$SERVED_NAME"  \
  --port               "$PORT"         \
  --host               "0.0.0.0"       \
  --max-model-len      "$MAX_LEN"      \
  --gpu-memory-utilization "$GPU_MEM"  \
  --quantization       bitsandbytes    \
  --dtype              float16         \
  --trust-remote-code                  \
  --enable-prefix-caching
