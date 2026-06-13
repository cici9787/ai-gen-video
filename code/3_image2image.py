#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调用 gpt-image-2 图生图（9:16）。"""

import argparse
import base64
import contextlib
import io
import os
import re

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1024x1824"  # 9:16，宽高均可被 16 整除
REF_IMG = "../data/定妆照/pic/女二阴险版.png"
PROMPT_FILE = "../data/定妆照/prompts/tmp.txt"
OUT_DIR = "../data/定妆照/pic"
OUT_NAME = "tmp1.png"

REF_PREFIX = (
    "【角色参考】严格参照附件参考图。若提供多张参考图：第一张是身份与服装主参考，"
    "必须保持同一张脸、同一发型与气质、同一外套、围巾、裤子、靴子等服装识别特征；"
    "后续参考图只用于辅助表情、情绪或氛围，不得改变第一张参考图的服装。"
    "仅按以下提示词调整，不要换脸。\n\n"
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


def fix_size(size):
    size = str(size).strip().lower().replace("×", "x").replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size)
    if not m:
        raise ValueError("尺寸格式应为：宽x高，例如 1024x1824")
    w, h = int(m.group(1)), int(m.group(2))
    return f"{w // 16 * 16}x{h // 16 * 16}"


def load_prompt(here, prompt_file):
    path = os.path.join(here, prompt_file)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def save_png(b64_data, out_path):
    from PIL import Image
    data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def call_api(client, prompt, size, ref_img_paths, use_prefix=True):
    prompt = clean_prompt(prompt)
    if use_prefix:
        prompt = clean_prompt(REF_PREFIX + prompt)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in ref_img_paths]
        image_arg = files[0] if len(files) == 1 else files
        resp = client.images.edit(
            model=MODEL, prompt=prompt, size=size, n=1, image=image_arg,
        )
    return resp.data[0].b64_json


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="gpt-image-2 图生图")
    parser.add_argument("--image", nargs="+", default=[REF_IMG], help=f"参考图路径，可传多张，默认 {REF_IMG}")
    parser.add_argument("--prompt-file", default=PROMPT_FILE, help=f"提示词文件，默认 {PROMPT_FILE}")
    parser.add_argument("--prompt", default=None, help="直接传入提示词，优先级高于 --prompt-file")
    parser.add_argument("--output", default=None, help=f"输出路径，默认 {OUT_DIR}/{OUT_NAME}")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1024x1824")
    parser.add_argument("--no-prefix", action="store_true", help="不在提示词前加角色参考前缀")
    args = parser.parse_args()

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    ref_imgs = [
        img if os.path.isabs(img) else os.path.join(here, img)
        for img in args.image
    ]
    missing = [img for img in ref_imgs if not os.path.isfile(img)]
    if missing:
        raise SystemExit(f"参考图不存在：{missing[0]}")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    size = fix_size(args.size)
    out_path = args.output or os.path.join(here, OUT_DIR, OUT_NAME)
    prompt = args.prompt or load_prompt(here, args.prompt_file)

    print("图生图生成中……")
    for idx, ref_img in enumerate(ref_imgs, 1):
        print(f"  参考图{idx}：{ref_img}")
    print(f"  输出：{out_path}")
    b64 = call_api(client, prompt, size, ref_imgs, use_prefix=not args.no_prefix)
    save_png(b64, out_path)
    print(f"✓ 已保存：{out_path}")


if __name__ == "__main__":
    main()
