export CUDA_VISIBLE_DEVICES=6,7

vllm serve --model ./Pretrained_models/qwen3-vl-4B-ins \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 22002 \
  --max-num-seqs 5 \
  --max-model-len 16384 \
  --limit-mm-per-prompt.video 0 \
  --served-model-name qwen3-vl-4b \
  --gpu-memory-utilization 0.85 \
  --mm-processor-cache-gb 0 \