import os
import sys
import traceback

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/hf_cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/huggingface/datasets"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.argv = ["train", "train/deepseek_8b_qlora.yaml"]

if __name__ == "__main__":
    try:
        from llamafactory.train.tuner import run_exp
        run_exp()
    except SystemExit as e:
        print(f"\n[EXIT] code={e.code}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
