#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量图生图：为扫雪老人段落生成角色参考图（除女主外）。"""

import argparse
import base64
import contextlib
import io
import os
import re
import time

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1024x1824"  # 9:16 角色全身

PROMPT_DIR = "../data/扫雪老人_角色图提示词"
OUT_DIR = "../data/扫雪老人_角色图"
SCENE_REF = "../data/扫雪老人_场景图/01_暴雪小区楼下全景.png"
MAX_RETRIES = 3
RETRY_DELAY = 5

CHAR_REF_PREFIX = (
    "【角色生成】严格参照附件场景参考图，保持同一诺兰式末日胶片质感、"
    "同一低饱和冷蓝灰色调、同一暴雪氛围和居民楼风格，"
    "仅按以下提示词生成人物，不要换画风，不要变成纯场景空镜。\n\n"
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
    return re.sub(r"\s+", "", text)


def fix_size(size):
    size = str(size).strip().lower().replace("×", "x").replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size)
    if not m:
        raise ValueError("尺寸格式应为：宽x高，例如 1024x1824")
    w, h = int(m.group(1)), int(m.group(2))
    return f"{w // 16 * 16}x{h // 16 * 16}"


def load_md_prompt(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = re.findall(r"```text\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        raise ValueError(f"未找到提示词代码块：{path}")
    return strip_prompt_spaces("".join(blocks))


def save_png(b64_data, out_path):
    from PIL import Image
    data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def format_api_prompt(prompt):
    return clean_prompt(CHAR_REF_PREFIX + prompt)


def call_api(client, prompt, size, ref_img_paths):
    prompt = format_api_prompt(prompt)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in ref_img_paths]
        image_arg = files[0] if len(files) == 1 else files
        resp = client.images.edit(
            model=MODEL, prompt=prompt, size=size, n=1, image=image_arg,
        )
    return resp.data[0].b64_json


def generate_one_image(
    client, prompt, size, ref_img_paths, out_path,
    retries=MAX_RETRIES, retry_delay=RETRY_DELAY,
):
    for attempt in range(1, retries + 2):
        try:
            b64 = call_api(client, prompt, size, ref_img_paths)
            save_png(b64, out_path)
            return True
        except Exception as exc:
            if attempt > retries:
                print(f"  ✗ 失败：{out_path}")
                print(f"    已重试 {retries} 次，最后错误：{exc}")
                return False
            print(f"  ! 生成失败，{retry_delay} 秒后重试（第 {attempt}/{retries} 次）：{exc}")
            time.sleep(retry_delay)


def list_prompt_files(prompt_dir):
    files = sorted(f for f in os.listdir(prompt_dir) if f.endswith(".md"))
    return [os.path.join(prompt_dir, f) for f in files]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="扫雪老人段落角色图批量图生图")
    parser.add_argument("--prompt-dir", default=PROMPT_DIR, help="角色提示词目录")
    parser.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    parser.add_argument("--scene-ref", default=SCENE_REF, help="场景画风参考图")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1024x1824")
    parser.add_argument("--file", default=None, help="只生成指定 md 文件")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="失败重试次数")
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY, help="重试间隔秒数")
    parser.add_argument(
        "--no-chain-ref", action="store_true",
        help="禁用链式参考，始终只用场景参考图",
    )
    args = parser.parse_args()

    if args.retries < 0 or args.retry_delay < 0:
        raise SystemExit("--retries 和 --retry-delay 不能小于 0")

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量 MODEL_API_KEY 或 OPENAI_API_KEY")

    prompt_dir = os.path.join(here, args.prompt_dir)
    out_dir = os.path.join(here, args.out_dir)
    scene_ref = os.path.join(here, args.scene_ref)
    if not os.path.isfile(scene_ref):
        raise SystemExit(f"场景参考图不存在：{scene_ref}")

    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    files = list_prompt_files(prompt_dir)
    if args.file:
        files = [os.path.join(prompt_dir, args.file)]

    chain_ref = not args.no_chain_ref
    mode = "场景参考+链式角色参考" if chain_ref else "仅场景参考"
    print(f"共 {len(files)} 张角色图待生成，尺寸 {size}，模式 {mode}")
    print(f"场景参考图：{scene_ref}")

    ok, fail = 0, 0
    last_char_ref = None
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{name}.png")
        prompt = load_md_prompt(path)

        ref_paths = [scene_ref]
        if chain_ref and last_char_ref:
            ref_paths.append(last_char_ref)

        print(f"[{i}/{len(files)}] 生成中：{name}（{len(prompt)} 字符）")
        for idx, ref in enumerate(ref_paths, 1):
            print(f"  参考图{idx}：{ref}")

        if generate_one_image(client, prompt, size, ref_paths, out_path, args.retries, args.retry_delay):
            ok += 1
            last_char_ref = out_path
            print(f"  ✓ 已保存：{out_path}")
        else:
            fail += 1

    print(f"完成：成功 {ok} 张，失败 {fail} 张")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
