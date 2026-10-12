# -*- coding: utf-8 -*-
"""imgio.py — OpenCV 中文路径安全的读写封装（本项目铁律工具）

背景（2026-10-06 实测）：
    本机 cv2（4.11 / 5.0 均同）**无法打开路径含非 ASCII 字符的文件**：
      cv2.imread("...\\工具\\模型\\x.jpg")            -> None
      cv2.CascadeClassifier("...\\工具\\模型\\x.xml")  -> empty() == True 且抛 FileStorage 异常
    而本项目目录名全是中文（工具 / 模型 / 源素材 / 勇者无法休息 / 第N话），
    因此**任何直接传路径给 cv2 的代码都会静默失效**。

    典型事故：qcmod._cascade() 里 CascadeClassifier 抛异常被 except 吞掉 →
    faces 全空 → 人脸检测形同不存在 → 取景退化为纯像素判断（无任何报错）。

用法：
    from imgio import imread, imwrite, load_cascade
    im  = imread("...\\源素材\\第2话\\切片超清2x\\003.jpg")   # 不要用 cv2.imread
    cas = load_cascade("...\\工具\\模型\\lbpcascade_animeface.xml")
"""
import os
import tempfile

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图（中文路径安全）。失败返回 None。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img, ext=".jpg", params=None):
    """写图（中文路径安全）。返回 True/False。"""
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imread_gray(path):
    return imread(path, cv2.IMREAD_GRAYSCALE)


def load_cascade(xml_path):
    """加载级联分类器（中文路径安全）。

    cv2.CascadeClassifier 读不了非 ASCII 路径，这里先复制到 ASCII 临时文件再加载。
    返回 classifier；文件不存在或加载失败时 raise（**不要静默吞掉**）。
    """
    xml_path = str(xml_path)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    try:
        xml_path.encode("ascii")
        cas = cv2.CascadeClassifier(xml_path)
    except UnicodeEncodeError:
        tmp = os.path.join(tempfile.gettempdir(),
                           "cascade_%d.xml" % (abs(hash(xml_path)) % 10 ** 8))
        if (not os.path.exists(tmp)) or os.path.getmtime(tmp) < os.path.getmtime(xml_path):
            with open(xml_path, "rb") as f:
                data = f.read()
            with open(tmp, "wb") as f:
                f.write(data)
        cas = cv2.CascadeClassifier(tmp)
    if cas.empty():
        raise RuntimeError(f"级联模型加载失败（文件损坏或路径不可读）: {xml_path}")
    return cas
