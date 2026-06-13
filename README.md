# ai-gen-video

《末日极寒：地球之盐》短剧 AI 视频素材生成工具集。基于知乎内部 `gpt-image-2` 接口，批量生成分镜图、定妆照、场景图与角色图。

从 [MyBook](https://github.com/) 项目中的 `ai_gen_video` 模块独立拆分。

## 目录结构

```
ai-gen-video/
├── code/          # Python 脚本（在 code/ 目录下运行）
├── data/          # 提示词、分镜稿、参考图与生成输出
│   ├── 分镜/
│   ├── 定妆照/
│   ├── 关键场景/
│   └── 扫雪老人_*/
└── requirements.txt
```

## 环境准备

```bash
cd ai-gen-video
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_API_KEY=sk-你的key
# 或 export OPENAI_API_KEY=sk-你的key
```

## 脚本说明

在 `code/` 目录下执行（脚本内路径相对 `../data/`）：

| 脚本 | 用途 |
|------|------|
| `1_shot_images.py` | 从分镜稿批量生成分镜关键帧图 |
| `2_text2image.py` | 文生图（读取提示词文件） |
| `2_extract_video_frames.py` | 从视频提取首尾帧对比 |
| `3_image2image.py` | 图生图 |
| `4_granny_yulan_sweater.py` | 903 老太太 + 玉兰毛衣定妆 |
| `5_female2_sinister_yulan_sweater.py` | 女二阴险版 + 玉兰毛衣 |
| `6_female2_bend_reveal.py` | 女二弯腰露毛衣 |
| `7_snow_scene_images.py` | 扫雪场景图批量生成 |
| `8_snow_scene_characters.py` | 扫雪角色图批量生成 |
| `example/text2image.py` | 文生图通用示例 |
| `example/image2image.py` | 图生图通用示例 |

### 示例

```bash
cd code

# 为镜头 34 生成 6 张分镜图
python 1_shot_images.py --shot 34

# 批量生成扫雪场景图
python 7_snow_scene_images.py

# 文生图试跑
python example/text2image.py --example
```

## 接口说明

默认使用知乎 Model API：

- Base URL: `https://model.in.zhihu.com/v1/`
- Model: `gpt-image-2`

## License

Private / 个人创作项目素材工具，请勿泄露 API Key。
