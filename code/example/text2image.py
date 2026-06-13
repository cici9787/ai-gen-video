#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 gpt-image-2 文生图脚本。

直接试跑（无参数时默认使用内置示例）：
    export MODEL_API_KEY=你的key
    python text2image.py

自定义生成：
    python text2image.py --prompt "写实电影感，暴雪城市空镜" --output ./output.png
"""

import argparse
import base64
import io
import os
import re
import time

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
DEFAULT_SIZE = "1024x1024"
MAX_RETRIES = 3
RETRY_DELAY = 5

PROMPT_MARKERS = (
    "【主提示词·直接复制】",
    "【主提示词·独立道具参考图】",
    "【主提示词】",
    "## 场景图提示词",
)

# 内置示例：无人物、无文字、无敏感信息，可直接 --example 试跑
EXAMPLE_PROMPT = (
    "16:9横屏，写实电影感，北方城市居民小区楼下暴雪空镜，"
    "九层带电梯居民楼，冷灰蓝色外墙，厚积雪覆盖道路，"
    "白毛风横刮，低饱和冷蓝灰色调，街道空寂，无人物"
)
EXAMPLE_NEGATIVE = (
    "不要字幕，不要可读文字，不要人物，不要卡通，不要3D，"
    "不要水印，不要温暖阳光，不要插画感"
)
EXAMPLE_OUTPUT = "example_output.png"


def clean_key(key):
    for ch in "\"\"''\u2018\u2019\u201c\u201d":
        key = key.replace(ch, "")
    return key.strip()


def clean_prompt(text):
    return (
        text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
    )


def strip_prompt_spaces(text):
    return re.sub(r"\s+", "", text)


def fix_size(size):
    size = str(size).strip().lower().replace("×", "x").replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size)
    if not m:
        raise ValueError("尺寸格式应为：宽x高，例如 1024x1024")
    w, h = int(m.group(1)), int(m.group(2))
    return f"{w // 16 * 16}x{h // 16 * 16}"


def load_prompt_from_md(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = re.findall(r"```text\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.strip()


def load_prompt_from_txt(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    main = None
    for marker in PROMPT_MARKERS:
        if marker in text:
            main = text.split(marker, 1)[1].split("\n---", 1)[0].strip()
            break
    if main is None:
        main = text

    if "【负面提示词】" in text:
        neg = text.split("【负面提示词】", 1)[1].split("\n---", 1)[0].strip()
        main = f"{main}\n{neg}"

    return main


def load_prompt(path):
    if path.lower().endswith(".md"):
        return load_prompt_from_md(path)
    return load_prompt_from_txt(path)


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def save_png(b64_data, out_path):
    from PIL import Image
    data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def call_api(client, prompt, size):
    prompt = clean_prompt(prompt)
    resp = client.images.generate(model=MODEL, prompt=prompt, size=size, n=1)
    return resp.data[0].b64_json


def generate_image(client, prompt, size, out_path, retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    for attempt in range(1, retries + 2):
        try:
            b64 = call_api(client, prompt, size)
            save_png(b64, out_path)
            return True
        except Exception as exc:
            if attempt > retries:
                print(f"生成失败：{out_path}")
                print(f"已重试 {retries} 次，最后错误：{exc}")
                return False
            print(f"生成失败，{retry_delay} 秒后重试（第 {attempt}/{retries} 次）：{exc}")
            time.sleep(retry_delay)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="gpt-image-2 通用文生图")
    parser.add_argument(
        "--example", action="store_true",
        help=f"使用内置示例提示词试跑，输出到当前目录 {EXAMPLE_OUTPUT}",
    )
    parser.add_argument("--prompt", default=None, help="直接传入提示词文本")
    parser.add_argument("--prompt-file", default=None, help="提示词文件路径，支持 .txt / .md")
    parser.add_argument("--output", default=None, help="输出图片路径，例如 ./output.png")
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"图片尺寸，默认 {DEFAULT_SIZE}")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="失败重试次数，默认 3")
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY, help="重试间隔秒数，默认 5")
    parser.add_argument(
        "--keep-spaces", action="store_true",
        help="保留提示词中的空格和换行；默认会合并空白字符",
    )
    args = parser.parse_args()

    use_example = args.example or (not args.prompt and not args.prompt_file)
    if use_example:
        if args.prompt or args.prompt_file:
            raise SystemExit("--example 不能与 --prompt / --prompt-file 同时使用")
        args.prompt = f"{EXAMPLE_PROMPT}{EXAMPLE_NEGATIVE}"
        args.output = args.output or EXAMPLE_OUTPUT
        if args.size == DEFAULT_SIZE:
            args.size = "1824x1024"
    elif args.prompt and args.prompt_file:
        raise SystemExit("--prompt 与 --prompt-file 不能同时使用")
    elif not args.output:
        raise SystemExit("请提供 --output")
    if args.retries < 0:
        raise SystemExit("--retries 不能小于 0")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay 不能小于 0")

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量 MODEL_API_KEY 或 OPENAI_API_KEY")

    if args.prompt:
        prompt = args.prompt.strip()
    else:
        prompt_path = resolve_path(args.prompt_file, here)
        if not os.path.isfile(prompt_path):
            raise SystemExit(f"提示词文件不存在：{prompt_path}")
        prompt = load_prompt(prompt_path)

    if not args.keep_spaces:
        prompt = strip_prompt_spaces(prompt)

    out_path = resolve_path(args.output, os.getcwd())
    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    print("文生图生成中……")
    if use_example:
        print("模式：内置示例（默认）")
    print(f"模型：{MODEL}")
    print(f"尺寸：{size}")
    print(f"提示词长度：{len(prompt)} 字符")
    print(f"输出：{out_path}")

    if generate_image(
        client, prompt, size, out_path,
        retries=args.retries, retry_delay=args.retry_delay,
    ):
        print(f"已保存：{out_path}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
