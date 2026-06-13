#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用公司 gpt-image-2 接口生成分镜图。

提示词 = 精简风格前缀 + 镜头「分镜提示词」+ 当前时间点关键帧说明（自动截断过长内容）。

单张模式（默认）：为指定镜头生成 N 张连续关键帧图（默认 6 张），保存到同一个文件夹。
批量模式（--multi）：按「视频运动提示词」子镜头生成，每个子镜头 N 张连续细分关键帧。
全部镜头（--all）：对分镜稿中每个镜头执行上述逻辑。
检查模式（--dry-run）：只打印每张图的提示词，不调用图片接口。
"""

import argparse
import base64
import io
import os
import re
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ── 配置 ────────────────────────────────────────────────────────────
MODEL    = "gpt-image-2"
BASE_URL = "https://model.in.zhihu.com/v1/"
MD_FILE  = "../data/分镜/分镜2.md"     # 分镜稿路径
REF_IMG  = "../data/定妆照/pic/女主角色图.jpeg"  # 默认参考图（仅锁定角色1）
OUT_DIR  = "../data/分镜/pic"           # 输出目录

# 角色图字段到定妆照文件的映射；参考图接口只上传角色1，角色2靠文字描述生成
CHAR_REF_MAP = {
    "女主": "../data/定妆照/pic/女主角色图.jpeg",
    "女主角色图": "../data/定妆照/pic/女主角色图.jpeg",
    "女主定妆照参考": "../data/定妆照/pic/女主角色图.jpeg",
    "司机": "../data/定妆照/pic/司机.png",
    "司机师傅": "../data/定妆照/pic/司机.png",
}
# SIZE     = "1024x1536" # 2:3，比较稳
SIZE     = "1024x1824" # 接近 9:16，且宽高都能被 16 整除
# SIZE = "1080x1920"

# 发给接口时，在提示词前面加上参考图说明；有角色2时会动态补充该角色描述
REF_PREFIX = "参照参考图，保持角色1同一张脸，仅改场景、服装和动作。\n\n"

# 精简风格前缀，避免和分镜提示词重复堆叠
STYLE_PREFIX = (
    "9:16竖屏单张电影分镜图，手绘彩铅质感，单画面不分格，无文字无字幕，"
    "自然光线，冷蓝灰雪景色调，室内暖肤色，红色玉兰毛衣可作点缀。"
)

# 提示词长度上限（字符数），过长时自动截断
MAX_SHOT_PROMPT_CHARS = 500
MAX_SUB_SHOT_CHARS = 520
MAX_FIELD_CHARS = 180
MAX_TIMELINE_ITEM_CHARS = 90

# 分镜默认时长；分镜稿里没有「时长」字段时使用
DEFAULT_SHOT_DURATION = 10.0


# ── 工具函数 ─────────────────────────────────────────────────────────

def clean_key(key):
    """去掉从网页复制 API Key 时混入的弯引号和空格，防止请求头报错。"""
    for ch in "\"\"''\u2018\u2019\u201c\u201d":
        key = key.replace(ch, "")
    return key.strip()


def clean_prompt(text):
    """把弯引号换成直引号，防止网关拒绝。"""
    return (
        text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
    )


def fix_size(size):
    """接口要求宽高都能被 16 整除；不满足时自动向下修正。"""
    size = str(size).strip().lower().replace("×", "x").replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size)
    if not m:
        raise ValueError("尺寸格式应为：宽x高，例如 1024x1536")

    w, h = int(m.group(1)), int(m.group(2))
    fixed_w, fixed_h = w // 16 * 16, h // 16 * 16
    fixed = f"{fixed_w}x{fixed_h}"

    if fixed != size:
        print(f"尺寸 {size} 不能被 16 整除，已自动改为 {fixed}")
    return fixed


def read_md(md_path):
    """读取分镜 md 全文。"""
    with open(md_path, encoding="utf-8") as f:
        return f.read()


def list_shot_numbers(md_path):
    """列出分镜稿中所有镜头编号。"""
    content = read_md(md_path)
    nums = [int(n) for n in re.findall(r"^## 镜头 (\d+)[｜|]", content, re.MULTILINE)]
    if not nums:
        raise ValueError(f"在 {md_path} 里没找到任何镜头")
    return sorted(nums)


def read_shot_section(md_path, shot_num):
    """从 md 文件里截取某个镜头的整段内容。"""
    content = read_md(md_path)
    m = re.search(
        rf"## 镜头 {shot_num}[｜|].*?(?=## 镜头 |\Z)",
        content, re.DOTALL,
    )
    if not m:
        raise ValueError(f"在 {md_path} 里没找到「镜头 {shot_num}」")
    return m.group(0)


def shorten_text(text, max_len):
    """压缩空白并截断过长文本。"""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def build_storyboard_prompt(section, extra=""):
    """拼接精简风格前缀 + 分镜提示词 + 可选子镜头内容。"""
    parts = [STYLE_PREFIX]
    shot_prompt = get_field(section, "分镜提示词")
    if shot_prompt:
        parts.append(shorten_text(shot_prompt, MAX_SHOT_PROMPT_CHARS))
    if extra:
        parts.append(shorten_text(extra, MAX_SUB_SHOT_CHARS))
    return "\n\n".join(parts)


def build_keyframe_prompt(section, frame_idx, total_frames, sub_shots=None):
    """为第 frame_idx 张图生成只包含当前画面的 image2 提示词。"""
    duration = get_shot_duration(section)
    frame_start, frame_end = frame_time_range(frame_idx, total_frames, duration)
    frame_mid = (frame_start + frame_end) / 2
    current = pick_sub_shot(sub_shots, frame_mid)

    if current:
        frame_text = current["desc"]
    else:
        frame_text = build_fallback_frame_text(section, frame_idx, total_frames)

    return build_image_prompt(section, frame_text, frame_idx, total_frames, frame_start, frame_end)


def build_image_prompt(section, frame_text, frame_idx, total_frames, frame_start, frame_end):
    """拼接最终发送给图片接口的提示词，只描述当前单张画面。"""
    parts = [
        STYLE_PREFIX,
        "只生成这一张单画面，不要多格漫画、不要故事板排版、不要画面编号、不要文字字幕。",
        f"当前画面是同一 10 秒镜头中的第 {frame_idx}/{total_frames} 张连续关键帧，"
        f"约 {format_seconds(frame_start)}-{format_seconds(frame_end)}，只表现这个时间点的画面。",
    ]

    character_prompt = build_character_prompt(section)
    if character_prompt:
        parts.append(character_prompt)

    scene_prompt = build_scene_prompt(section)
    if scene_prompt:
        parts.append(scene_prompt)

    parts.append("当前画面内容：" + shorten_text(frame_text, MAX_SUB_SHOT_CHARS))
    parts.append("连续性要求：主体身份、服装、道具、空间关系、镜头方向和光线保持稳定；动作只推进当前这一小段，不要混入其他时间点的物品或动作。")
    parts.append("画面限制：不要出现可读文字、字幕、品牌 logo、楼号、车牌号、UI 文本或夸张摆拍。")

    char2 = get_character(section, 2)
    if char2 and character_is_visible(char2["desc"]):
        parts.append(
            f"双人场景要求：{char2['name']}必须作为独立人物出现，"
            "不要把他画成女主，也不要把女主画成司机；两人位置、动作和表情各自独立。"
        )

    return "\n\n".join(parts)


def get_character(section, idx):
    """读取单个角色的名称、描述和定妆照标记。"""
    name = get_field(section, f"角色{idx}")
    desc = get_field(section, f"角色描述{idx}")
    ref_tag = get_field(section, f"角色图{idx}")
    if not name or name == "无":
        return None
    return {
        "name": name,
        "desc": desc if desc and desc != "无" else "",
        "ref_tag": ref_tag if ref_tag and ref_tag != "无" else "",
    }


def character_is_visible(desc):
    """判断角色2是否会在画面里直接出现。"""
    if not desc:
        return False
    hidden_markers = ("不露面", "不直接出现", "不出现在", "只通过", "只以", "画外")
    return not any(marker in desc for marker in hidden_markers)


def build_ref_prefix(section):
    """根据当前镜头角色，生成发送给 image2 的参考图前缀。"""
    char1 = get_character(section, 1)
    if not char1:
        return REF_PREFIX

    lines = [
        f"参照参考图，保持{char1['name']}同一张脸，仅改场景、服装和动作。",
        "参考图只锁定角色1，不要把参考图人物误画成其他角色。",
    ]

    char2 = get_character(section, 2)
    if char2 and char2["desc"]:
        if character_is_visible(char2["desc"]):
            ref_hint = ""
            if char2["ref_tag"]:
                ref_hint = f"，外貌气质参考{char2['ref_tag']}定妆照"
            lines.append(
                f"{char2['name']}不在参考图中，请按文字单独生成{ref_hint}："
                f"{shorten_text(char2['desc'], MAX_FIELD_CHARS)}"
            )
            lines.append(
                f"画面中必须同时区分{char1['name']}和{char2['name']}，"
                f"两人服装、年龄、性别和站位不要混淆。"
            )
        else:
            lines.append(
                f"{char2['name']}不直接露面，只通过{shorten_text(char2['desc'], MAX_FIELD_CHARS)}表现，"
                "画面中不要画出其完整正脸。"
            )

    return "\n".join(lines) + "\n\n"


def build_character_prompt(section):
    """提取当前镜头需要保留的人物设定，区分参考图角色与文字生成角色。"""
    parts = []
    char1 = get_character(section, 1)
    char2 = get_character(section, 2)

    if char1 and char1["desc"]:
        parts.append(f"{char1['name']}（参考图锁定）：{shorten_text(char1['desc'], MAX_FIELD_CHARS)}")

    if char2 and char2["desc"]:
        if character_is_visible(char2["desc"]):
            ref_hint = f"，参考{char2['ref_tag']}定妆照" if char2["ref_tag"] else ""
            parts.append(
                f"{char2['name']}（文字生成{ref_hint}）：{shorten_text(char2['desc'], MAX_FIELD_CHARS)}"
            )
        else:
            parts.append(f"{char2['name']}（不露面）：{shorten_text(char2['desc'], MAX_FIELD_CHARS)}")

    return "人物设定：" + "；".join(parts) if parts else ""


def build_scene_prompt(section):
    """提取光影和情绪约束，不引入完整剧情时间轴。"""
    parts = []
    for field_name in ("场景标签", "光影氛围", "情绪"):
        value = get_field(section, field_name)
        if value:
            parts.append(f"{field_name}：{shorten_text(value, MAX_FIELD_CHARS)}")
    return "\n".join(parts)


def get_field(section, field_name):
    """从镜头段落里提取某个字段（**字段名** 下面的文本）。"""
    m = re.search(
        rf"\*\*{field_name}\*\*\s*\n(.+?)(?=\n\*\*|\n---|\Z)",
        section, re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def get_sub_shots(section):
    """
    从「视频运动提示词」里拆出每个子镜头。
    子镜头格式：镜头1（0:00-0:03）：……
    返回 [{"idx": 1, "start": 0.0, "end": 2.0, "desc": "..."}]
    """
    video_text = get_field(section, "视频运动提示词")
    if not video_text:
        raise ValueError("该镜头缺少「视频运动提示词」字段")

    # 拆出每个子镜头（不再拼接视频运动提示词里那段超长风格说明）
    subs = re.findall(
        r"镜头\s*(\d+)[（(]([^）)]*)[）)]\s*[：:]\s*(.+?)(?=\n镜头\s*\d+[（(]|\Z)",
        video_text, re.DOTALL,
    )
    if not subs:
        raise ValueError("「视频运动提示词」里没找到子镜头（格式：镜头1（时间）：...）")

    result = []
    for i, time_text, desc in subs:
        start, end = parse_time_range(time_text)
        result.append({
            "idx": int(i),
            "start": start,
            "end": end,
            "desc": clean_prompt(desc.strip()),
        })
    return result


def get_shot_duration(section):
    """从「时长」字段提取秒数，失败时默认 10 秒。"""
    duration_text = get_field(section, "时长")
    m = re.search(r"(\d+(?:\.\d+)?)\s*秒", duration_text)
    return float(m.group(1)) if m else DEFAULT_SHOT_DURATION


def parse_time_value(text):
    """解析 0:02 / 02 / 2秒 这类时间值为秒。"""
    text = str(text).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return int(minutes) * 60 + float(seconds)
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else 0.0


def parse_time_range(text):
    """解析子镜头括号里的时间范围。"""
    parts = re.split(r"\s*[-—~～至到]\s*", text.strip(), maxsplit=1)
    if len(parts) != 2:
        return 0.0, DEFAULT_SHOT_DURATION
    start = parse_time_value(parts[0])
    end = parse_time_value(parts[1])
    if end <= start:
        end = start + 1.0
    return start, end


def frame_time_range(frame_idx, total_frames, duration):
    """把 N 张图均匀映射到分镜时长。"""
    start = duration * (frame_idx - 1) / total_frames
    end = duration * frame_idx / total_frames
    return start, end


def pick_sub_shot(sub_shots, current_time):
    """按当前时间点匹配所属子镜头。"""
    if not sub_shots:
        return None
    for sub in sub_shots:
        if sub["start"] <= current_time < sub["end"]:
            return sub
    return sub_shots[-1]


def format_seconds(value):
    """把秒数格式化为简短中文时间。"""
    value = round(float(value), 1)
    if value.is_integer():
        return f"{int(value)}秒"
    return f"{value:g}秒"


def format_timeline(sub_shots):
    """把子镜头压缩成完整时间轴摘要，帮助模型理解前后顺序。"""
    lines = []
    for sub in sub_shots:
        lines.append(
            f"- {format_seconds(sub['start'])}-{format_seconds(sub['end'])}："
            f"{shorten_text(sub['desc'], MAX_TIMELINE_ITEM_CHARS)}"
        )
    return "\n".join(lines)


def build_fallback_frame_text(section, frame_idx, total_frames):
    """没有视频运动提示词时，用当前序号提示模型只画对应阶段。"""
    shot_prompt = get_field(section, "分镜提示词")
    action = get_field(section, "角色动作")
    if frame_idx == 1:
        stage = "镜头开端阶段，动作刚开始，情绪刚建立。"
    elif frame_idx == total_frames:
        stage = "镜头收尾阶段，动作完成或情绪落点明确。"
    else:
        stage = "镜头中段推进阶段，动作和情绪处于连续变化中。"
    return "；".join(
        item for item in (
            stage,
            shorten_text(action, MAX_FIELD_CHARS) if action else "",
            shorten_text(shot_prompt, MAX_SHOT_PROMPT_CHARS) if shot_prompt else "",
        )
        if item
    )


def save_png(b64_data, out_path):
    """把 base64 图片数据保存为 PNG 文件。"""
    from PIL import Image
    data = base64.b64decode(b64_data)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def format_api_prompt(prompt, ref_img=None, section=None):
    """返回实际发送给接口的完整提示词。"""
    prompt = clean_prompt(prompt)
    if ref_img:
        prefix = build_ref_prefix(section) if section else REF_PREFIX
        return clean_prompt(prefix + prompt)
    return prompt


def print_prompt(prompt, ref_img=None, section=None):
    """打印单张图片实际发送给接口的提示词。"""
    api_prompt = format_api_prompt(prompt, ref_img, section)
    print(f"  ── 提示词（{len(api_prompt)} 字）──")
    print(api_prompt)
    print("  ── 提示词结束 ──")


def generate_one_image(client, prompt, size, ref_img, out_path, retries, retry_delay, section=None):
    """生成单张图；失败时重试，超过次数后返回 False。"""
    for attempt in range(1, retries + 2):
        try:
            b64 = call_api(client, prompt, size, ref_img, section)
            save_png(b64, out_path)
            return True
        except Exception as exc:
            if attempt > retries:
                print(f"  ✗ 失败：{out_path}")
                print(f"    已重试 {retries} 次，最后错误：{exc}")
                return False

            print(f"  ! 生成失败，{retry_delay} 秒后重试（第 {attempt}/{retries} 次）：{exc}")
            time.sleep(retry_delay)


def output_folder(args, out_dir, shot_num, sub_shot_num=None):
    """生成输出文件夹。"""
    if args.output:
        base_dir = args.output
    else:
        base_dir = os.path.join(out_dir, f"镜头{shot_num}")

    if sub_shot_num is not None:
        return os.path.join(base_dir, f"图{sub_shot_num}")
    return base_dir


def generate_multi(client, section, args, size, ref_img, out_dir):
    """批量模式：每个子镜头生成 count 张连续细分关键帧。"""
    sub_shots = get_sub_shots(section)
    total = len(sub_shots) * args.count
    success = 0
    failed = 0
    print(f"批量模式：镜头 {args.shot}，共 {len(sub_shots)} 个子镜头 × {args.count} 张连续细分关键帧 = {total} 张，开始生成……")
    for sub in sub_shots:
        folder = output_folder(args, out_dir, args.shot, sub["idx"])
        for n in range(1, args.count + 1):
            prompt = build_sub_shot_keyframe_prompt(section, sub_shots, sub, n, args.count)
            out_path = os.path.join(folder, f"第{n}张.png")
            print(f"\n  [子镜头{sub['idx']} 第{n}/{args.count}张] 生成中……")
            print_prompt(prompt, ref_img, section)
            if args.dry_run:
                print(f"  dry-run：不调用接口；原本会保存到：{out_path}")
                success += 1
                continue
            if generate_one_image(
                client, prompt, size, ref_img, out_path,
                args.retries, args.retry_delay, section,
            ):
                success += 1
                print(f"  ✓ 已保存：{out_path}")
            else:
                failed += 1
    print(f"\n完成！镜头 {args.shot} 成功 {success} 张，失败 {failed} 张。")


def generate_single(client, section, args, size, ref_img, out_dir):
    """单张模式：为镜头生成 count 张按 10 秒时间轴推进的连续关键帧。"""
    folder = output_folder(args, out_dir, args.shot)
    success = 0
    failed = 0
    try:
        sub_shots = get_sub_shots(section)
    except ValueError as exc:
        print(f"提示：{exc}，将使用画面描述/角色动作生成顺序关键帧。")
        sub_shots = []

    print(f"连续关键帧模式：镜头 {args.shot}，生成 {args.count} 张可按顺序串联的分镜图……")
    for n in range(1, args.count + 1):
        prompt = build_keyframe_prompt(section, n, args.count, sub_shots)
        out_path = os.path.join(folder, f"第{n}张.png")
        print(f"\n  [第{n}/{args.count}张] 生成中……")
        print_prompt(prompt, ref_img, section)
        if args.dry_run:
            print(f"  dry-run：不调用接口；原本会保存到：{out_path}")
            success += 1
            continue
        if generate_one_image(
            client, prompt, size, ref_img, out_path,
            args.retries, args.retry_delay, section,
        ):
            success += 1
            print(f"  ✓ 已保存：{out_path}")
        else:
            failed += 1
    print(f"\n完成！镜头 {args.shot} 成功 {success} 张，失败 {failed} 张。")


def build_sub_shot_keyframe_prompt(section, sub_shots, sub, frame_idx, total_frames):
    """为 --multi 模式生成子镜头内部的连续细分关键帧提示词。"""
    duration = sub["end"] - sub["start"]
    local_start, local_end = frame_time_range(frame_idx, total_frames, duration)
    frame_start = sub["start"] + local_start
    frame_end = sub["start"] + local_end
    frame_mid = (frame_start + frame_end) / 2

    frame_text = (
        f"子镜头{sub['idx']}内部第 {frame_idx}/{total_frames} 张细分关键帧，"
        f"中心时间点约 {format_seconds(frame_mid)}。"
        f"{shorten_text(sub['desc'], MAX_SUB_SHOT_CHARS)}"
    )
    return build_image_prompt(section, frame_text, frame_idx, total_frames, frame_start, frame_end)


def call_api(client, prompt, size, ref_img_path=None, section=None):
    """
    调用接口生成图片，返回 base64 字符串。
    - ref_img_path=None → 纯文生图（/images/generations）
    - ref_img_path=文件路径 → 图生图（/images/edits）
    """
    prompt = format_api_prompt(prompt, ref_img_path, section)
    if ref_img_path:
        with open(ref_img_path, "rb") as f:
            resp = client.images.edit(
                model=MODEL, prompt=prompt, size=size, n=1, image=f,
            )
    else:
        resp = client.images.generate(
            model=MODEL, prompt=prompt, size=size, n=1,
        )
    return resp.data[0].b64_json


# ── 主程序 ───────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="gpt-image-2 分镜图生成工具")
    parser.add_argument("--shot",   type=int, default=12,      help="镜头编号，默认 12（与 --all 互斥）")
    parser.add_argument("--all",    action="store_true",       help="处理分镜稿中的全部镜头")
    parser.add_argument("--output", default=None,              help="输出文件夹（不填则自动命名）")
    parser.add_argument("--size",   default=SIZE,              help="图片尺寸，例如 1024x1536、1024x1024")
    parser.add_argument("--no-ref", action="store_true",       help="不用参考图，纯文生图")
    parser.add_argument("--multi",  action="store_true",       help="批量模式：按子镜头生成多张图")
    parser.add_argument("--count",  type=int, default=6,       help="每个分镜/子镜头生成的连续关键帧数量，默认 6")
    parser.add_argument("--retries", type=int, default=3,      help="单张图片失败后的重试次数，默认 3")
    parser.add_argument("--retry-delay", type=int, default=5,  help="每次重试前等待秒数，默认 5")
    parser.add_argument(
        "--dry-run", "--print-only",
        action="store_true",
        help="只打印每张图的最终提示词和输出路径，不调用图片接口",
    )
    args = parser.parse_args()

    if args.all and args.output:
        raise SystemExit("--all 模式下不能指定 --output，请使用自动命名")
    if args.retries < 0:
        raise SystemExit("--retries 不能小于 0")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay 不能小于 0")
    if args.count <= 0:
        raise SystemExit("--count 必须大于 0")

    client = None
    if not args.dry_run:
        # API Key
        if OpenAI is None:
            raise SystemExit("当前 Python 环境缺少 openai 包；只检查提示词可使用 --dry-run")

        raw_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        api_key = clean_key(raw_key)
        if not api_key:
            raise SystemExit("请先设置环境变量：export MODEL_API_KEY=sk-你的key")

        client = OpenAI(base_url=BASE_URL, api_key=api_key)
    size    = fix_size(args.size)
    md_path = os.path.join(here, MD_FILE)
    ref_img = None if args.no_ref else os.path.join(here, REF_IMG)
    out_dir = os.path.join(here, OUT_DIR)
    shot_nums = list_shot_numbers(md_path) if args.all else [args.shot]

    if args.dry_run:
        print("dry-run 模式：只打印提示词，不调用图片接口，不保存图片。")
    print(f"待处理镜头：{shot_nums}")

    for shot_num in shot_nums:
        args.shot = shot_num
        section = read_shot_section(md_path, shot_num)
        if args.multi:
            generate_multi(client, section, args, size, ref_img, out_dir)
        else:
            generate_single(client, section, args, size, ref_img, out_dir)


if __name__ == "__main__":
    main()


'''
# 为镜头12生成6张连续关键帧图，保存到 ../data/分镜/pic/镜头12/
python 1_shot_images.py --shot 12

# 只打印镜头12的6张连续关键帧提示词，不调用图片接口
python 1_shot_images.py --shot 12 --dry-run

# 为全部分镜各生成6张连续关键帧图，每个镜头一个文件夹
python 1_shot_images.py --all

# 生成分镜17的子镜头细分关键帧，每个子镜头生成6张，保存到镜头17/图1、镜头17/图2...
python 1_shot_images.py --shot 17 --multi

# 纯文生图，不用参考图
python 1_shot_images.py --shot 17 --no-ref
'''
