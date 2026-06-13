#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文生图生成 902 女主家全景（独立脚本，自然室内打光）。"""

import os

from utils.image_gen import (
    fix_size,
    get_client,
    load_sections,
    resolve_path,
    save_png,
    strip_prompt_spaces,
    text_to_image,
    with_retry,
)

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT_FILE = resolve_path("../data/关键场景/prompts/S1_902女主家全景.txt", HERE)
OUT_PATH = resolve_path("../data/关键场景/场景图/S1_902女主家全景.png", HERE)

# 帐篷造型+朝向置顶
TENT_PREFIX = (
    "【帐篷朝向·必遵守】床2米长边贴墙暖气，帐篷长轴与2米长边平行；"
    "帐门只在床尾端朝衣柜半开，露出帐内睡袋。绝对禁止帐门朝客厅左侧、朝餐桌、"
    "朝入户门、朝床宽侧面。高清写实8K。\n\n"
)

# 打光单独写在脚本里，只拼主提示词（空间/陈设），避免与 txt 里多层光影描述叠加出怪光
LIGHT_PREFIX = (
    "【打光】8K超清写实室内摄影，照片级高清细节，暴雪夜但电力暖气正常。"
    "客厅厨房天花板吸顶灯暖白均匀照明，全屋可读、无死黑；"
    "结霜窗外仅在窗边有极淡青白冷反光，不要光柱、不要丁达尔光、不要上帝光；"
    "卧室无窗，比客厅略暗半档；帐篷与暖气片旁仅极弱暖反光，帐篷不发光、帐内不金黄。"
    "低饱和自然色调，明暗过渡柔和，禁止HDR、禁止舞台追光、禁止一半金一半蓝。"
    "【画面】16:9超广角一镜全屋：玄关门、厨房、客厅窗与餐桌、卧室帐篷衣柜全部入画。\n\n"
)


def build_prompt():
    body = load_sections(PROMPT_FILE, "主提示词")
    return strip_prompt_spaces(TENT_PREFIX + LIGHT_PREFIX + body)


def main():
    client = get_client()
    prompt = build_prompt()
    size = fix_size("1824x1024")
    print(f"文生图：女主家全景（{size}）→ {OUT_PATH}")
    b64 = with_retry(lambda: text_to_image(client, prompt, size))
    save_png(b64, OUT_PATH)
    print(f"✓ 已保存：{OUT_PATH}")


if __name__ == "__main__":
    main()
