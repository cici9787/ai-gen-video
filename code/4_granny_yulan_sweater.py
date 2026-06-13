#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用老太太参考图 + 玉兰毛衣参考图，生成903老太太定妆照（内穿玉兰毛衣 + 外搭长款羽绒）。"""

import argparse
import base64
import contextlib
import io
import os
import re

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1024x1824"  # 9:16，适合全身角色定妆

GRANNY_IMG = "../data/定妆照/pic/903老太太.jpeg"
SWEATER_IMG = "../data/定妆照/pic/玉兰毛衣.png"
PROMPT_FILE = "../data/定妆照/prompts/03_903老太太.txt"
OUT_DIR = "../data/定妆照/pic"
OUT_NAME = "903老太太.png"

REF_PREFIX = """
【参考图说明——最高优先级】
第一张参考图：903老太太身份参考。必须严格保持她的脸型、五官、花白短发、年龄感、慈祥气质、真实皱纹和身体比例，不要换脸、不要年轻化。
第二张参考图：已定稿玉兰毛衣。内层必须穿这件同款毛衣——鲜艳酒红、及膝长款、胸前玉兰枝刺绣、底边横向玉兰边饰，颜色版型针织质感与参考图一致。

【生成任务】
按下方定妆提示词，生成老太太全身定妆照：内层穿第二张参考图的玉兰毛衣，外层套明亮藏青色（中明度清爽蓝调，比深藏青更亮、不发黑不发灰）厚实长款羽绒服，不要暗沉、不要紫色、不要黑色、不要灰色；羽绒衣摆仍盖过红毛衣下摆，羽绒服拉链拉开、前襟自然敞开露出红毛衣和胸前玉兰刺绣；双手自然垂放于身体两侧，不要拉扯衣襟、不要举手摆拍。

""".strip()


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
    path = prompt_file if os.path.isabs(prompt_file) else os.path.join(here, prompt_file)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def save_png(b64_data, out_path):
    from PIL import Image
    data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def call_api(client, prompt, size, image_paths):
    prompt = clean_prompt(prompt)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in image_paths]
        resp = client.images.edit(
            model=MODEL,
            prompt=prompt,
            size=size,
            n=1,
            image=files,
        )
    return resp.data[0].b64_json


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="生成903老太太定妆照（玉兰毛衣+长款羽绒）")
    parser.add_argument("--granny", default=GRANNY_IMG, help=f"老太太身份参考图，默认 {GRANNY_IMG}")
    parser.add_argument("--sweater", default=SWEATER_IMG, help=f"玉兰毛衣参考图，默认 {SWEATER_IMG}")
    parser.add_argument("--prompt-file", default=PROMPT_FILE, help=f"提示词文件，默认 {PROMPT_FILE}")
    parser.add_argument("--output", default=None, help=f"输出路径，默认 {OUT_DIR}/{OUT_NAME}")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1024x1824")
    parser.add_argument("--no-prefix", action="store_true", help="不在提示词前加参考图说明前缀")
    args = parser.parse_args()

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    granny = args.granny if os.path.isabs(args.granny) else os.path.join(here, args.granny)
    sweater = args.sweater if os.path.isabs(args.sweater) else os.path.join(here, args.sweater)
    image_paths = [granny, sweater]
    missing = [path for path in image_paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"参考图不存在：{missing[0]}")

    out_path = args.output or os.path.join(here, OUT_DIR, OUT_NAME)
    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    prompt_body = load_prompt(here, args.prompt_file)
    prompt = prompt_body if args.no_prefix else f"{REF_PREFIX}\n\n{prompt_body}"

    print("903老太太定妆照生成中……")
    print(f"  身份参考图：{granny}")
    print(f"  毛衣参考图：{sweater}")
    print(f"  提示词文件：{args.prompt_file}")
    print(f"  输出：{out_path}")
    b64 = call_api(client, prompt, size, image_paths)
    save_png(b64, out_path)
    print(f"✓ 已保存：{out_path}")


if __name__ == "__main__":
    main()
