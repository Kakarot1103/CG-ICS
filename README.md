# Toward Robust In-Context Segmentation via Concept Guidance

**Zhigang Chen**, **Xiawu Zheng**, **Rongrong Ji**

📣 **Accepted to ECCV 2026**

[[`Paper (arXiv)`](https://arxiv.org/abs/2606.28149)]

<div align="center">
  <img src="https://img.shields.io/badge/ECCV-2026-blue">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green">
</div>

This repository is the **official implementation** of *"Toward Robust In-Context Segmentation via Concept Guidance"* (CG-ICS). CG-ICS is a training-free framework that combines **SAM3** with an **MLLM** (Qwen3-VL) for text-concept extraction and tree-search prompt selection, making in-context segmentation robust across diverse benchmarks.

---

## 📋 Table of Contents

- [Data Preparation](#-data-preparation)
- [Environment](#-environment)
- [Model Preparation](#-model-preparation)
- [Running](#-running)
- [Citation](#-citation)

---

## 📦 Data Preparation

The datasets (PASCAL, COCO, FSS, LVIS, PASCAL-Part, ISIC, iSAID) follow the standard in-context segmentation data layout. Please refer to the [**GF-SAM datasets README**](https://github.com/ANDYZAQ/GF-SAM/blob/master/datasets/README.md) for how to download and organize them.

After downloading, place (or symlink) the data so that the repository contains a `data/` directory pointing to your dataset root, e.g.:

```bash
ln -s /path/to/your/datasets data
```

---

## 🌱 Environment

Tested with **Python 3.12** and **CUDA 12.8** on NVIDIA RTX 3090 GPUs.

1. Create a conda environment **named `sam3`** (this name is required — `scripts/all_serial.sh` activates a conda env called `sam3`):

   ```bash
   conda create -n sam3 python=3.12 -y
   conda activate sam3
   ```

2. Then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** If you prefer a different environment name, update `CONDA_ENV_NAME="sam3"` near the top of `scripts/all_serial.sh` accordingly.

---

## 🧠 Model Preparation

This repository uses two pretrained models: **SAM3** (the segmenter) and **Qwen3-VL** (the MLLM used for concept extraction). Please download both and place them under a `Pretrained_models/` directory at the repository root.

```bash
mkdir -p Pretrained_models
cd Pretrained_models
```

- **SAM3.** Download the SAM3 checkpoint (`sam3.pt`) following the [**SAM3 repository**](https://github.com/facebookresearch/sam3), then place it here as `Pretrained_models/sam3.pt`.
- **Qwen3-VL.** Download the Qwen3-VL weights (we use `Qwen3-VL-4B-Instruct`) following the [**Qwen3-VL repository**](https://github.com/QwenLM/qwen3-vl), then place them here as `Pretrained_models/qwen3-vl-4B-ins/`.

The expected layout is:

```
Pretrained_models/
├── sam3.pt
└── qwen3-vl-4B-ins/        # the Qwen3-VL model directory (config.json, *.safetensors, ...)
```

### Deploying Qwen3-VL with vLLM

We deploy Qwen3-VL as an OpenAI-compatible API server with [vLLM](https://github.com/vllm-project/vllm), running on **2× NVIDIA RTX 3090** GPUs via tensor parallelism:

```bash
bash scripts/llm_server.sh
```

This serves the model on `http://localhost:22002/v1` with `--served-model-name qwen3-vl-4b`.

---

## 🚀 Running

Make sure the vLLM server (`scripts/llm_server.sh`) is running before launching evaluation.

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --benchmark coco --fold 0 --nshot 1 --datapath data \
  --save_dir ./results/coco_fold0_1shot
```

### Multi-GPU (DDP via `torchrun`)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --benchmark coco --fold 0 --nshot 1 --datapath data \
  --save_dir ./results/coco_fold0_1shot
```

The example above uses `coco`, `fold 0`, `1-shot`; replace `--benchmark`/`--fold`/`--nshot` to run other settings (`--benchmark` choices: `fss, coco, pascal, lvis, pascal_part, isic, isaid`).

### Running multiple benchmarks/folds

To run several benchmarks and folds serially (each as a DDP job) inside a tmux session, use the provided harness:

```bash
bash scripts/all_serial.sh
```

---

## 📑 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{chen2026robustincontextsegmentationconcept,
      title={Toward Robust In-Context Segmentation via Concept Guidance},
      author={Zhigang Chen and Xiawu Zheng and Rongrong Ji},
      year={2026},
      eprint={2606.28149},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.28149},
}
```

---

## 📄 License

This repository is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).

The `sam3/` directory is vendored third-party source from Meta's [SAM3](https://github.com/facebookresearch/sam3) and is governed by its **own SAM License** ([`sam3/LICENSE`](sam3/LICENSE)), which takes precedence within that directory. By using the `sam3/` code you agree to comply with the terms of that license.
