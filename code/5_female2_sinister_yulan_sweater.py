#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用女二阴险版参考图 + 玉兰毛衣参考图，生成女二阴险版角色图（内穿玉兰毛衣、直立不露馅）。"""

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

SINISTER_IMG = "../data/定妆照/pic/女二2阴险版.png"  # 输入参考：脸、表情、气质（固定）
SWEATER_IMG = "../data/定妆照/pic/玉兰毛衣.png"
PROMPT_FILE = "../data/定妆照/prompts/02_女二_阴险版.txt"
OUT_DIR = "../data/定妆照/pic"
OUT_NAME = "女二阴险版角色图.png"  # 输出定妆图（不与输入同名）

STYLE_IMG = "../data/定妆照/pic/女二阴险版角色图.png"  # 可选：已定稿服装方向；默认不启用

REF_PREFIX = """
【参考图说明——最高优先级】
第一张参考图：女二阴险版身份锚点（pic/女二2阴险版.png）。必须严格保持同一张脸、阴险恶毒黑化表情、发型、身材、黑裤黑靴与反派气质，不要换脸。不要参考其错误内搭红毛衣、及小腿长羽绒或敞开穿法。
第二张参考图：已定稿 pic/玉兰毛衣.png。内层 1:1 及膝长款玉兰毛衣，穿在里面被羽绒完全藏住。
{style_note}

【生成任务】
脸与表情以第一张为准；内层定稿玉兰毛衣；外层宝石墨绿膝上羽绒、克制宽腰带、少量高级结构缝线、无围巾、拉链全闭、直立零红色。白底直立定妆。

""".strip()

STYLE_NOTE = "第三张参考图：当前已定稿服装方向（pic/女二阴险版角色图.png）。只参考其宝石墨绿配色、膝上衣长、宽腰带与整体版型方向，不要参考其可能错误的条纹或细节；脸与表情仍以第一张为准。"


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
    parser = argparse.ArgumentParser(description="生成女二阴险版角色图（阴险版参考+玉兰毛衣）")
    parser.add_argument("--sinister", default=SINISTER_IMG, help=f"阴险版参考图（输入），默认 {SINISTER_IMG}")
    parser.add_argument("--style", default="none", help=f"服装方向参考，默认 none；可传 {STYLE_IMG}")
    parser.add_argument("--sweater", default=SWEATER_IMG, help=f"玉兰毛衣参考，默认 {SWEATER_IMG}")
    parser.add_argument("--prompt-file", default=PROMPT_FILE, help=f"提示词文件，默认 {PROMPT_FILE}")
    parser.add_argument("--output", default=None, help=f"输出路径，默认 {OUT_DIR}/{OUT_NAME}")
    parser.add_argument("--size", default=SIZE, help="图片尺寸，默认 1024x1824")
    parser.add_argument("--no-prefix", action="store_true", help="不在提示词前加参考图说明前缀")
    args = parser.parse_args()

    api_key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

    sinister = args.sinister if os.path.isabs(args.sinister) else os.path.join(here, args.sinister)
    sweater = args.sweater if os.path.isabs(args.sweater) else os.path.join(here, args.sweater)
    out_path = args.output or os.path.join(here, OUT_DIR, OUT_NAME)

    if os.path.normpath(sinister) == os.path.normpath(out_path):
        raise SystemExit("输入参考图与输出不能是同一文件；输入请用 女二2阴险版.png，输出为 女二阴险版角色图.png")

    image_paths = [sinister, sweater]
    if args.style and args.style.lower() != "none":
        style = args.style if os.path.isabs(args.style) else os.path.join(here, args.style)
        if not os.path.isfile(style):
            raise SystemExit(f"服装参考图不存在：{style}")
        if os.path.normpath(style) == os.path.normpath(out_path):
            print("  提示：服装参考与输出同名，已跳过（避免输入输出同一文件）")
        else:
            image_paths.append(style)

    missing = [path for path in image_paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"参考图不存在：{missing[0]}")
    size = fix_size(args.size)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    prompt_body = load_prompt(here, args.prompt_file)
    style_note = STYLE_NOTE if len(image_paths) > 2 else ""
    prefix = REF_PREFIX.format(style_note=style_note) if not args.no_prefix else ""
    prompt = prompt_body if args.no_prefix else f"{prefix}\n\n{prompt_body}"

    print("女二阴险版角色图生成中……")
    print(f"  输入参考图：{sinister}")
    if len(image_paths) > 2:
        print(f"  服装参考图：{image_paths[2]}")
    print(f"  毛衣参考图：{sweater}")
    print(f"  提示词文件：{args.prompt_file}")
    print(f"  输出：{out_path}")
    b64 = call_api(client, prompt, size, image_paths)
    save_png(b64, out_path)
    print(f"✓ 已保存：{out_path}")


if __name__ == "__main__":
    main()
