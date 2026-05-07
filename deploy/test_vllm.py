"""Phase 4: vLLM 部署集成测试

测试项目:
  1. 基础对话    — 验证模型可正常响应
  2. 流式输出    — 验证 WebSocket 流式通路
  3. 多轮对话    — 验证上下文连贯性
  4. 危机关键词  — 验证模型在危机场景下是否回复正常 (不崩溃)
  5. 并发请求    — 验证 vLLM 并发处理能力

用法:
  python deploy/test_vllm.py
  python deploy/test_vllm.py --url http://gpu-server:8001/v1 --model deepseek-8b-qlora
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "\033[92m[通过]\033[0m"
FAIL = "\033[91m[失败]\033[0m"
SKIP = "\033[93m[跳过]\033[0m"


async def test_basic(client, model: str) -> bool:
    """基础对话：验证可正常响应"""
    print("测试 1: 基础对话")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业的心理健康咨询师。"},
                {"role": "user", "content": "我最近工作压力很大，总是睡不好觉，该怎么办？"},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        print(f"  响应长度: {len(content)} 字")
        print(f"  预览: {content[:80]}...")
        if len(content) < 50:
            print(f"  {FAIL} 响应过短")
            return False
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


async def test_streaming(client, model: str) -> bool:
    """流式输出：验证 token 逐步推送"""
    print("\n测试 2: 流式输出")
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业的心理健康咨询师。"},
                {"role": "user", "content": "我感觉很焦虑，能帮帮我吗？"},
            ],
            temperature=0.4,
            max_tokens=256,
            stream=True,
        )
        token_count = 0
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token_count += 1
        print(f"  收到 token 数: {token_count}")
        if token_count < 10:
            print(f"  {FAIL} 流式 token 过少")
            return False
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


async def test_multi_turn(client, model: str) -> bool:
    """多轮对话：验证上下文连贯"""
    print("\n测试 3: 多轮对话")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业的心理健康咨询师。"},
                {"role": "user", "content": "我最近情绪很低落"},
                {"role": "assistant", "content": "我听到你了，能告诉我是什么让你感到低落吗？"},
                {"role": "user", "content": "感觉做什么都没意思，提不起劲"},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        print(f"  响应长度: {len(content)} 字")
        if len(content) < 30:
            print(f"  {FAIL} 多轮响应过短")
            return False
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


async def test_crisis_safety(client, model: str) -> bool:
    """危机场景：验证模型不会崩溃，且回复正常"""
    print("\n测试 4: 危机关键词场景")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业的心理健康咨询师。"},
                {"role": "user", "content": "我感觉活着很累，有时候觉得消失了也无所谓"},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        print(f"  响应长度: {len(content)} 字")
        if len(content) < 20:
            print(f"  {FAIL} 危机场景响应过短")
            return False
        print(f"  {PASS} (热线注入由 MentalAgent 代码层保证)")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


async def test_concurrency(client, model: str) -> bool:
    """并发测试：5 个并发请求"""
    print("\n测试 5: 并发请求 (5 个)")
    msgs = [
        {"role": "system", "content": "你是心理健康咨询师。"},
        {"role": "user", "content": "帮我放松一下"},
    ]

    async def one_req(idx: int) -> float:
        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=model, messages=msgs, max_tokens=64, temperature=0.5,
        )
        return time.monotonic() - t0

    try:
        t0 = time.monotonic()
        latencies = await asyncio.gather(*[one_req(i) for i in range(5)])
        total = time.monotonic() - t0
        avg = sum(latencies) / len(latencies)
        print(f"  并发耗时: {total:.1f}s | 平均单请求: {avg:.1f}s")
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


async def main(url: str, model: str) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="EMPTY", base_url=url)

    # 连接检查
    print(f"连接 vLLM: {url}")
    try:
        models = await client.models.list()
        names = [m.id for m in models.data]
        print(f"可用模型: {names}")
        if model not in names:
            if names:
                print(f"[提示] '{model}' 不在列表，使用第一个: {names[0]}")
                model = names[0]
            else:
                print("[错误] 没有可用模型")
                sys.exit(1)
    except Exception as e:
        print(f"[错误] 无法连接 vLLM 服务: {e}")
        print(f"  请确认 vLLM 已启动: bash deploy/start_vllm.sh")
        sys.exit(1)

    print()
    tests = [
        test_basic(client, model),
        test_streaming(client, model),
        test_multi_turn(client, model),
        test_crisis_safety(client, model),
        test_concurrency(client, model),
    ]
    results = await asyncio.gather(*[t for t in tests[:4]])  # 前 4 个顺序执行
    results += (await asyncio.gather(tests[4]),)             # 并发测试最后

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 45}")
    print(f"  测试结果: {passed}/{total} 通过")
    print(f"{'=' * 45}")

    if passed == total:
        print(f"\n  vLLM 部署验证成功！")
        print(f"  确认 .env 中已设置: VLLM_BASE_URL={url}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM 部署验证")
    parser.add_argument("--url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="deepseek-8b-qlora")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.model))
