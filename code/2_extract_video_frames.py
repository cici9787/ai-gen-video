#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract first and last frames from video files."""

import argparse
from pathlib import Path

import imageio.v2 as imageio


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "video_promt" / "frames_compare"
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent.parent / "video_promt"


def save_image(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def read_last_frame(reader):
    """Scan to the last frame. This is slower but reliable across codecs."""
    last = None
    for frame in reader:
        last = frame
    return last


def extract_frames(video_path: Path, output_dir: Path) -> None:
    reader = imageio.get_reader(video_path)
    try:
        first = reader.get_data(0)
        last = read_last_frame(reader)
    finally:
        reader.close()

    if last is None:
        raise RuntimeError(f"Cannot read last frame: {video_path}")

    save_image(output_dir / f"{video_path.stem}_first.png", first)
    save_image(output_dir / f"{video_path.stem}_last.png", last)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first and last frames from videos.")
    parser.add_argument("videos", nargs="*", help="Video file paths; default: ../video_promt/*.mp4")
    args = parser.parse_args()
    print("111")

    videos = [Path(video) for video in args.videos] or sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {DEFAULT_VIDEO_DIR}")

    for video in videos:
        extract_frames(video, DEFAULT_OUTPUT_DIR)
        print(f"Extracted: {video} -> {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

'''
1.环境安装：pip install imageio imageio-ffmpeg
2.抽帧：python extract_video_frames.py
3.视频动作提示词：有了首尾帧之后，丢给大模型让它去想出一个衔接的视频提示词。
4.衔接视频生成：可以使用vidu里的图生视频，上传首位帧和视频提示词，生成视频。
'''