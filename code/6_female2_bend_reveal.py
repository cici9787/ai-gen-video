#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成女二阴险版「弯腰关903门露出长款玉兰毛衣」子视图。"""

import argparse
import base64
import contextlib
import io
import os
import re

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
SIZE = "1536x1024"  # 16:9，适合楼道弯腰场景

SINISTER_IMG = "../data/定妆照/pic/女二2阴险版.png"
SWEATER_IMG = "../data/定妆照/pic/玉兰毛衣.png"
SCENE_REF = "../data/关键场景/参考图/S6_伪楼长露出玉兰毛衣_参考.png"
PROMPT_FILE = "../data/定妆照/prompts/02_女二_阴险版_弯腰子视图.txt"
OUT_DIR = "../data/定妆照/pic"
OUT_NAME = "女二阴险版_弯腰露毛衣.png"

REF_PREFIX = """
【参考图说明——最高优先级】
第一张参考图：女二阴险版。保持同一张脸、发型、身材与反派气质，不要换脸。
第二张参考图：已定稿 pic/玉兰毛衣.png。内层长款毛衣必须 1:1——鲜艳酒红、圆领无高领、修身及膝长款、底边非对称横向玉兰边饰；弯腰时从大腿下段到膝部露出这段红毛衣下摆，底边玉兰边饰必须清晰醒目。
第三张参考图：S6 楼道弯腰关门场景构图参考。参考走廊氛围、903 门、弯腰关门动作与机位，但人物脸和服装按前两张与下方提示词执行；内层毛衣花型以第二张定稿为准，不要参考第三张里错误的开衫/胸花。

【生成任务】
生成女二阴险版【弯腰关 903 门】子视图：伪楼长弯腰关门，短款黑羽绒前摆上翻，内层及膝长款玉兰毛衣从大腿下段到膝部露出，底边横向玉兰边饰清晰可见、红底白花像物证；楼道冷蓝暗光。这是直立定妆照的配套子视图，验证「正常站立不露、弯腰才露」。

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
        raise ValueError("尺寸格式应为：宽x高，例如 1536x1024")
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
    parser = argparse.ArgumentParser(description="生成女二阴险版弯腰露毛衣子视图")
    parser.add_argument("--sinister", default=SINISTER_IMG)
    parser.add_argument("--sweater", default=SWEATER_IMG)
    parser.add_argument("--scene", default=SCENE_REF, help="场景构图参考，传 none 跳过")
    parser.add_argument("--prompt-file", default=PROMPT_FILE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--size", default=SIZE)
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args()

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    sinister = args.sinister if os.path.isabs(args.sinister) else os.path.join(here, args.sinister)
    sweater = args.sweater if os.path.isabs(args.sweater) else os.path.join(here, args.sweater)
    image_paths = [sinister, sweater]
    if args.scene and args.scene.lower() != "none":
        scene = args.scene if os.path.isabs(args.scene) else os.path.join(here, args.scene)
        if os.path.isfile(scene):
            image_paths.append(scene)

    missing = [path for path in image_paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"参考图不存在：{missing[0]}")

    out_path = args.output or os.path.join(here, OUT_DIR, OUT_NAME)
    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    prompt_body = load_prompt(here, args.prompt_file)
    prompt = prompt_body if args.no_prefix else f"{REF_PREFIX}\n\n{prompt_body}"

    print("女二阴险版弯腰露毛衣子视图生成中……")
    for idx, path in enumerate(image_paths, 1):
        print(f"  参考图{idx}：{path}")
    print(f"  提示词：{args.prompt_file}")
    print(f"  输出：{out_path}")
    b64 = call_api(client, prompt, size, image_paths)
    save_png(b64, out_path)
    print(f"✓ 已保存：{out_path}")


if __name__ == "__main__":
    main()
