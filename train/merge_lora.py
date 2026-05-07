"""Phase 4: 将 LoRA adapter 合并到基础模型，为 vLLM 部署做准备

两种方式:
  llamafactory (推荐) — 自动生成导出配置，交给 LLaMA-Factory 执行
  peft               — 纯 Python，不依赖 LLaMA-Factory

用法:
  # 推荐: 生成导出配置，再运行
  python train/merge_lora.py
  llamafactory-cli export train/export_config.yaml

  # 手动合并 (CPU，无需 GPU)
  python train/merge_lora.py --method peft --output models/mental_health_merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_export_config(base_model: str, adapter: str, output: str) -> Path:
    """生成 LLaMA-Factory 模型导出配置文件"""
    config = f"""### LLaMA-Factory 模型导出配置
### 运行命令: llamafactory-cli export train/export_config.yaml

model_name_or_path: {base_model}
adapter_name_or_path: {adapter}
template: qwen
finetuning_type: lora
export_dir: {output}
export_size: 4        # 分片大小 (GB)
export_device: cpu    # CPU 合并，避免 GPU OOM；3B 模型约需 6 GB 系统内存
export_legacy_format: false
"""
    config_path = Path(__file__).parent / "export_config.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path


def merge_with_peft(base_model: str, adapter: str, output: str) -> None:
    """使用 PEFT 手动合并权重 (不依赖 LLaMA-Factory)"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"加载基础模型 (CPU bfloat16，约需 6 GB 系统内存)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"加载 LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    print("合并权重...")
    merged = model.merge_and_unload()

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"保存合并模型: {out}")
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(out)

    print(f"\n合并完成，下一步:")
    print(f"  bash deploy/start_vllm.sh --model {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 LoRA 合并")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="outputs/mental_health_lora")
    parser.add_argument("--output", default="models/mental_health_merged")
    parser.add_argument(
        "--method",
        choices=["llamafactory", "peft"],
        default="llamafactory",
        help="合并方式 (llamafactory=推荐, peft=手动)",
    )
    args = parser.parse_args()

    if not Path(args.adapter).exists():
        print(f"错误: adapter 路径不存在: {args.adapter}", file=sys.stderr)
        print("请先完成训练: llamafactory-cli train train/deepseek_8b_qlora.yaml")
        sys.exit(1)

    if args.method == "llamafactory":
        config_path = generate_export_config(args.base_model, args.adapter, args.output)
        print(f"导出配置已生成: {config_path}")
        print(f"运行: llamafactory-cli export {config_path}")
    else:
        merge_with_peft(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
