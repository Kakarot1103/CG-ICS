#!/usr/bin/env python3
"""Command line entry point that mirrors the original sam3_agent notebook."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.model.sam3_image_processor import Sam3Processor


SAM3_EXAMPLES_DIR = Path(__file__).resolve().parent
SAM3_ROOT = SAM3_EXAMPLES_DIR.parent

LLM_CONFIGS = {
    # vLLM-served models
    "qwen3_vl_8b_thinking": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
    },
    # extend here with API-based providers if needed
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SAM 3 Agent example outside of a Jupyter notebook."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(SAM3_ROOT / "assets" / "images" / "test_image.jpg"),
        help="待分割图像路径，默认使用示例图片。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="the leftmost child wearing blue vest",
        help="需要 SAM 3 Agent 解析的文本提示。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SAM3_ROOT / "agent_output"),
        help="保存结果图像的目录。",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3_vl_8b_thinking",
        help="需要使用的多模态大模型标识，应当存在于 LLM_CONFIGS 中。",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=os.environ.get("LLM_API_KEY", "DUMMY_API_KEY"),
        help="调用 LLM 所需的 API key。",
    )
    parser.add_argument(
        "--llm-server-url",
        type=str,
        default="http://0.0.0.0:8001/v1",
        help="vLLM 服务地址，若使用其他 provider 可覆盖 base_url。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("bfloat16", "float16", "none"),
        default="bfloat16",
        help="CUDA autocast 精度，不需要 autocast 可选择 none。",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES 环境变量的取值。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启后会保存更多中间调试信息。",
    )
    return parser.parse_args()


def enable_torch_optimizations(precision: str) -> ExitStack:
    """Setup torch wide contexts for inference."""
    stack = ExitStack()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if precision != "none":
            dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
            try:
                stack.enter_context(torch.autocast("cuda", dtype=dtype))
            except RuntimeError as err:
                print(f"[WARN] autocast 失败 ({err}), 将继续执行但不启用 autocast。")
        else:
            print("[INFO] autocast 已禁用。")
    else:
        print("[INFO] 未检测到 CUDA，可在 CPU 或其他设备上运行。")
    stack.enter_context(torch.inference_mode())
    return stack


def build_processor() -> Sam3Processor:
    sam3_root = Path(sam3.__file__).resolve().parent.parent
    bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    return Sam3Processor(model, confidence_threshold=0.5)


def prepare_llm_config(
    model_name: str, api_key: str, server_url: str
) -> Tuple[dict, Optional[str]]:
    if model_name not in LLM_CONFIGS:
        available = ", ".join(LLM_CONFIGS)
        raise ValueError(f"未知的模型 {model_name}，可选项：{available}")
    llm_config = LLM_CONFIGS[model_name].copy()
    llm_config["name"] = model_name
    llm_config["api_key"] = api_key
    if "model" not in llm_config:
        llm_config["model"] = model_name
    if llm_config.get("provider") == "vllm":
        return llm_config, server_url
    llm_config.setdefault("base_url", server_url)
    return llm_config, llm_config["base_url"]


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # notebook 中的逻辑会切换到 sam3 目录，这里保持一致，避免相对路径混乱
    os.chdir(SAM3_ROOT)

    llm_config, server_url = prepare_llm_config(
        args.llm_model, args.llm_api_key, args.llm_server_url
    )
    processor = build_processor()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (SAM3_ROOT / image_path).resolve()
    prompt = args.prompt

    send_generate_request = partial(
        send_generate_request_orig,
        server_url=server_url,
        model=llm_config["model"],
        api_key=llm_config["api_key"],
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 即将处理图像: {image_path}")
    with enable_torch_optimizations(args.precision):
        output_image_path = run_single_image_inference(
            str(image_path),
            prompt,
            llm_config,
            send_generate_request,
            call_sam_service,
            debug=args.debug,
            output_dir=str(output_dir),
        )

    if output_image_path is None:
        print("[WARN] 未获得分割结果。")
        sys.exit(1)

    print(f"[INFO] 推理完成，结果图像已保存到: {output_image_path}")


if __name__ == "__main__":
    main()
