# -*- coding: utf-8 -*-
"""douyin_compose.py — 4:3 主成片 → 抖音 9:16（1080x1920）模糊背景合成。
零决策纯模板：底层=放大铺满+高斯模糊+压暗；前景=4:3 满宽、中心在画高 44%；
顶部富余区可选标题钩子（PIL 中文字体）。

★关键教训（已实证）：PyAV 里「视频写完再追加音轨」会产生损坏的 mp4
（InvalidDataError 1094995529）。必须单遍双流：一次性 inp.decode() 同时拿到
视频帧和音频帧，视频帧合成后重编 libx264，音频帧重采样为 fltp/stereo/44100
后重编 AAC，边解边 mux 交错写入。

用法：
  D:\\develop\\python\\python.exe douyin_compose.py <主成片.mp4> <输出.mp4> [--title 标题文字]

依赖：PyAV + cv2 + numpy + PIL（本机已装，无需 ffmpeg）。
"""
import argparse
import av, cv2, numpy as np
from av.video.frame import VideoFrame
from PIL import Image, ImageDraw, ImageFont

CW, CH = 1080, 1920          # 抖音画布
FG_W, FG_H = 1080, 810       # 前景 4:3 满宽
FG_Y = int(CH * 0.44) - FG_H // 2   # 中心略偏上（画高 44%），给底部 UI 让位
BLUR = 51                    # 高斯核（奇数）
DARK = 0.55                  # 背景压暗系数
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def draw_title(img, title):
    """PIL 画中文标题：黑描边白字，画高 8% 处。"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT_PATH, 64)
    for dx in (-3, 0, 3):
        for dy in (-3, 0, 3):
            d.text((40 + dx, int(CH * 0.08) + dy), title, font=f, fill=(0, 0, 0))
    d.text((40, int(CH * 0.08)), title, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--title", default=None, help="顶部标题钩子")
    a = ap.parse_args()

    inp = av.open(a.src)
    vs = inp.streams.video[0]
    fps = vs.average_rate or 30
    has_audio = bool(inp.streams.audio)

    out = av.open(a.out, "w")
    ovs = out.add_stream("libx264", rate=fps)
    ovs.width, ovs.height, ovs.pix_fmt = CW, CH, "yuv420p"
    ovs.options = {"crf": "18", "preset": "medium"}

    if has_audio:
        oas = out.add_stream("aac", rate=44100)
        oas.layout = "stereo"
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=44100)

    n_v = n_a = 0
    for frame in inp.decode():   # 单遍双流：视频帧 + 音频帧交错到达
        if isinstance(frame, av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            # 底层：放大铺满 1920 高 → 居中裁 1080 宽 → 模糊压暗
            bg = cv2.resize(img, (int(CH * img.shape[1] / img.shape[0]), CH))
            x0 = (bg.shape[1] - CW) // 2
            bg = bg[:, x0:x0 + CW]
            bg = (cv2.GaussianBlur(bg, (BLUR, BLUR), 0) * DARK).astype(np.uint8)
            # 前景：4:3 满宽，居中偏上
            bg[FG_Y:FG_Y + FG_H, 0:FG_W] = cv2.resize(img, (FG_W, FG_H))
            if a.title:
                bg = draw_title(bg, a.title)
            for pkt in ovs.encode(VideoFrame.from_ndarray(bg, format="bgr24")):
                out.mux(pkt)
            n_v += 1
        else:  # AudioFrame
            frame.pts = None
            for rf in resampler.resample(frame):
                for pkt in oas.encode(rf):
                    out.mux(pkt)
            n_a += 1

    # flush
    for pkt in ovs.encode():
        out.mux(pkt)
    if has_audio:
        for rf in resampler.resample(None):
            for pkt in oas.encode(rf):
                out.mux(pkt)
        for pkt in oas.encode():
            out.mux(pkt)

    inp.close()
    out.close()
    print(f"抖音版 -> {a.out}（{CW}x{CH}，视频帧 {n_v}，音频块 {n_a}）")


main()
