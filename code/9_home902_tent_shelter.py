#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图生图生成 902 帐篷避难所（参考女主家全景）。"""

import os

from utils.image_gen import (
    fix_size,
    get_client,
    image_to_image,
    load_prompt_file,
    resolve_path,
    save_png,
    strip_prompt_spaces,
    with_retry,
)

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT_TENT = resolve_path("../data/关键场景/prompts/S1_902女主家帐篷避难所.txt", HERE)
REF_PANORAMA = resolve_path("../data/关键场景/场景图/S1_902女主家全景.png", HERE)
OUT_TENT = resolve_path("../data/关键场景/场景图/S1_902女主家帐篷避难所.png", HERE)

TENT_REF_PREFIX = (
    "【图生图任务】把参考图床上深绿圆顶帐篷推进为帐内特写，保持同一材质色调。"
    "穹顶要更圆更饱满，半球形高拱；帐内打光更亮，暖白漫射光清晰可读。"
    "构图从睡袋腰腹中段取景不露头端；双人宽幅超级厚睡袋。"
    "左侧挂气罐和便携圆柱形露营灯（未点亮，不要台灯）；右侧挂防寒服。"
    "正前方帐门朝白墙，门缝外只有贴墙囤货物资，禁止门和柜子衣柜。"
    "9:16竖屏帐内低机位，8K高清写实照片级，不要人物，不要换画风。\n\n"
)

def main():
    if not os.path.isfile(REF_PANORAMA):
        raise SystemExit(f"缺少全景参考图：{REF_PANORAMA}\n请先运行：python 10_home902_panorama.py")

    client = get_client()
    prompt = strip_prompt_spaces(f"{TENT_REF_PREFIX}{load_prompt_file(PROMPT_TENT)}")
    size = fix_size("1024x1824")
    print(f"图生图：帐篷避难所（参考 {REF_PANORAMA}）→ {OUT_TENT}")
    b64 = with_retry(lambda: image_to_image(client, prompt, size, REF_PANORAMA))
    save_png(b64, OUT_TENT)
    print(f"✓ 已保存：{OUT_TENT}")


if __name__ == "__main__":
    main()
