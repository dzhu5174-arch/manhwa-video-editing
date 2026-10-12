# -*- coding: utf-8 -*-
"""cut_from_full 的 PyAV 版（无 ffmpeg 环境用）：按时间轴从整篇切句。
解码整篇 -> 采样级切片 -> libmp3lame 重编码，参数对齐原 ffmpeg 版（48k/24kHz/mono）。
"""
import av, csv, os, sys, argparse
import numpy as np


def read_pcm(path, rate=24000):
    c = av.open(path)
    rs = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks = []
    for fr in c.decode(c.streams.audio[0]):
        for r in rs.resample(fr):
            chunks.append(r.to_ndarray()[0])
    for r in rs.resample(None):
        chunks.append(r.to_ndarray()[0])
    return np.concatenate(chunks), rate


def write_mp3(path, pcm, rate=24000):
    out = av.open(path, "w")
    st = out.add_stream("libmp3lame", rate=rate)
    st.layout = "mono"
    st.bit_rate = 48000
    step = 1152 * 8
    for i in range(0, len(pcm), step):
        chunk = np.ascontiguousarray(pcm[i:i + step][None, :])
        f = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
        f.rate = rate
        for pkt in st.encode(f):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lead", type=float, default=0.05)
    ap.add_argument("--tail", type=float, default=0.10)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pcm, rate = read_pcm(a.full)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    if not rows:
        sys.exit("时间轴为空")
    for r in rows:
        tag = r["句号"].strip()
        st = max(0.0, float(r["起始秒"]) - a.lead)
        en = min(len(pcm) / rate, float(r["结束秒"]) + a.tail)
        write_mp3(os.path.join(a.out, f"L{tag}.mp3"), pcm[int(st * rate):int(en * rate)], rate)
    print(f"PyAV 切割完成：{len(rows)} 句 -> {a.out}")


if __name__ == "__main__":
    main()
