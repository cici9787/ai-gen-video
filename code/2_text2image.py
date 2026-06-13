#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调用 gpt-image-2 文生图生成玉兰毛衣道具参考图。"""

import argparse
import base64
import io
import os
import re

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1024x1024"  # 1:1，道具平铺参考图

PROMPT_FILE = "../data/定妆照/prompts/12_玉兰毛衣.txt"
OUT_DIR = "../data/定妆照/pic"
OUT_NAME = "玉兰毛衣.png"

PROMPT_MARKERS = (
    "【主提示词·直接复制】",
    "【主提示词·独立道具参考图】",
    "【主提示词】",
)


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
    """去掉提示词中的空格、换行和制表符，合并为连续文本。"""
    return re.sub(r"\s+", "", text)


def fix_size(size):
    size = str(size).strip().lower().replace("×", "x").replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size)
    if not m:
        raise ValueError("尺寸格式应为：宽x高，例如 1024x1024")
    w, h = int(m.group(1)), int(m.group(2))
    return f"{w // 16 * 16}x{h // 16 * 16}"


def load_prompt(here, prompt_file=PROMPT_FILE):
    path = os.path.join(here, prompt_file)
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
        main = main + neg

    return strip_prompt_spaces(main)


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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="gpt-image-2 玉兰毛衣道具参考图")
    parser.add_argument("--prompt-file", default=PROMPT_FILE, help=f"提示词文件，默认 {PROMPT_FILE}")
    parser.add_argument("--output", default=None, help=f"输出路径，默认 {OUT_DIR}/{OUT_NAME}")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1024x1024")
    args = parser.parse_args()

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    size = fix_size(args.size)
    out_path = args.output or os.path.join(here, OUT_DIR, OUT_NAME)
    prompt = load_prompt(here, args.prompt_file)

    print("玉兰毛衣道具参考图生成中……")
    print(f"提示词长度：{len(prompt)} 字符")
    b64 = call_api(client, prompt, size)
    save_png(b64, out_path)
    print(f"✓ 已保存：{out_path}")


if __name__ == "__main__":
    main()
