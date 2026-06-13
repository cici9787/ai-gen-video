#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量调用 gpt-image-2 为扫雪老人场景图提示词生成图片。"""

import argparse
import base64
import io
import os
import re
import time

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1824x1024"  # 16:9

PROMPT_DIR = "../data/扫雪老人_场景图提示词"
OUT_DIR = "../data/扫雪老人_场景图"
MAX_RETRIES = 3
RETRY_DELAY = 5

SCENE_REF_PREFIX = (
    "【场景参考】严格参照附件参考图，保持同一B市居民小区、同一九层及以上带电梯老小区建筑、"
    "同一2000年代建成但维护尚可、不破败的冷灰蓝外墙、"
    "同一深绿色单元门、同一诺兰式末日胶片质感、同一低饱和冷蓝灰色调和暴雪氛围，"
    "仅按以下提示词改变机位和场景内容，不要换成低层步梯楼、无电梯老破小，不要换小区风格，不要换画风。\n\n"
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
        raise ValueError("尺寸格式应为：宽x高，例如 1824x1024")
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


def format_api_prompt(prompt, ref_img_path=None):
    prompt = clean_prompt(prompt)
    if ref_img_path:
        return clean_prompt(SCENE_REF_PREFIX + prompt)
    return prompt


def call_api(client, prompt, size, ref_img_path=None):
    prompt = format_api_prompt(prompt, ref_img_path)
    if ref_img_path:
        with open(ref_img_path, "rb") as f:
            resp = client.images.edit(
                model=MODEL, prompt=prompt, size=size, n=1, image=f,
            )
    else:
        resp = client.images.generate(model=MODEL, prompt=prompt, size=size, n=1)
    return resp.data[0].b64_json


def generate_one_image(
    client, prompt, size, out_path,
    ref_img_path=None,
    retries=MAX_RETRIES, retry_delay=RETRY_DELAY,
):
    """生成单张图；失败时重试，超过次数后返回 False。"""
    for attempt in range(1, retries + 2):
        try:
            b64 = call_api(client, prompt, size, ref_img_path)
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
    parser = argparse.ArgumentParser(description="扫雪老人场景图批量文生图")
    parser.add_argument("--prompt-dir", default=PROMPT_DIR, help="提示词目录")
    parser.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1824x1024")
    parser.add_argument("--file", default=None, help="只生成指定 md 文件名，例如 01_暴雪小区楼下全景.md")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="单张图片失败后的重试次数，默认 3")
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY, help="每次重试前等待秒数，默认 5")
    parser.add_argument(
        "--no-chain-ref", action="store_true",
        help="禁用链式参考，每张都用纯文生图",
    )
    args = parser.parse_args()

    if args.retries < 0:
        raise SystemExit("--retries 不能小于 0")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay 不能小于 0")

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    prompt_dir = os.path.join(here, args.prompt_dir)
    out_dir = os.path.join(here, args.out_dir)
    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    files = list_prompt_files(prompt_dir)
    if args.file:
        files = [os.path.join(prompt_dir, args.file)]

    chain_ref = not args.no_chain_ref
    mode = "链式图生图（第2张起参考前一张）" if chain_ref else "纯文生图"
    print(f"共 {len(files)} 张场景图待生成，尺寸 {size}，模式 {mode}，失败最多重试 {args.retries} 次")

    ok, fail = 0, 0
    last_ref_path = None
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{name}.png")
        prompt = load_md_prompt(path)
        ref_img_path = last_ref_path if chain_ref else None

        print(f"[{i}/{len(files)}] 生成中：{name}（{len(prompt)} 字符）")
        if ref_img_path:
            print(f"  参考图：{ref_img_path}")
        else:
            print("  参考图：无（首张文生图）")

        if generate_one_image(
            client, prompt, size, out_path,
            ref_img_path=ref_img_path,
            retries=args.retries, retry_delay=args.retry_delay,
        ):
            ok += 1
            last_ref_path = out_path
            print(f"  ✓ 已保存：{out_path}")
        else:
            fail += 1

    print(f"完成：成功 {ok} 张，失败 {fail} 张")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
