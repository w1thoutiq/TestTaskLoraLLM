#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

cfg() {
  python -c 'import sys, yaml
v = yaml.safe_load(open("config.yaml", encoding="utf-8"))
for k in sys.argv[1].split("."): v = v[k]
print(v)' "$1"
}

exec python -m vllm.entrypoints.openai.api_server \
  --model "$(cfg model.id)" \
  --host "$(cfg serve.host)" --port "$(cfg serve.port)" \
  --dtype "$(cfg serve.dtype)" \
  --max-model-len "$(cfg serve.max_model_len)" \
  --gpu-memory-utilization "$(cfg serve.gpu_memory_utilization)" \
  --max-num-seqs "$(cfg serve.max_concurrent)" \
  --enable-lora \
  --lora-modules "$(cfg serve.adapter_name)=$(cfg paths.adapter)" \
  --max-loras "$(cfg serve.max_loras)" \
  --max-lora-rank "$(cfg serve.max_lora_rank)" \
  --enable-prefix-caching
