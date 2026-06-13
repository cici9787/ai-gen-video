#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpt-image-2 文生图 / 图生图公共工具。"""

import base64
import contextlib
import io
import os
import re
import time

from openai import OpenAI

MODEL = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"

PROMPT_MARKERS = ("【主提示词·直接复制】", "【主提示词】", "## 场景图提示词")
STOP_SECTIONS = ("【生成不理想时追加】", "【负面提示词】")


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


def resolve_path(path, base_dir):
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_api_key():
    key = clean_key(os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    if not key:
        raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")
    return key


def get_client():
    return OpenAI(base_url=BASE_URL, api_key=get_api_key())


def _trim_prompt(text):
    for marker in STOP_SECTIONS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def load_sections(path, *prefixes):
    """按【前缀】提取小节正文，标题可带后缀，如【整体风格——xxx】。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    parts = []
    for prefix in prefixes:
        pattern = rf"【{re.escape(prefix)}[^】]*】\s*\n(.*?)(?=\n【|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            parts.append(m.group(1).strip())
    return "\n\n".join(parts)


def load_prompt_file(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    for marker in PROMPT_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[1].split("\n---", 1)[0].strip()
            break

    return _trim_prompt(text)


def save_png(b64_data, out_path):
    from PIL import Image

    data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def text_to_image(client, prompt, size):
    prompt = clean_prompt(prompt)
    resp = client.images.generate(model=MODEL, prompt=prompt, size=size, n=1)
    return resp.data[0].b64_json


def image_to_image(client, prompt, size, ref_paths):
    prompt = clean_prompt(prompt)
    paths = [ref_paths] if isinstance(ref_paths, str) else list(ref_paths)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in paths]
        image_arg = files[0] if len(files) == 1 else files
        resp = client.images.edit(
            model=MODEL, prompt=prompt, size=size, n=1, image=image_arg,
        )
    return resp.data[0].b64_json


def with_retry(fn, retries=3, delay=5):
    for attempt in range(1, retries + 2):
        try:
            return fn()
        except Exception as exc:
            if attempt > retries:
                raise
            print(f"  ! 失败，{delay}s 后重试（{attempt}/{retries}）：{exc}")
            time.sleep(delay)
