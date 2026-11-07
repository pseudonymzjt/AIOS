# -*- coding: utf-8 -*-
"""用训练好的 TransNet 对自己的图片做风格迁移（推理，不训练）。

用法示例：
    # 默认用 4_2 的模型 models/fst.pth（udnie 风格）
    python style_transfer_demo.py ../..//my_photo.jpg

    # 指定模型（例如自己用 4_3 训练出来的 fst_train.pth）和输出目录
    python style_transfer_demo.py --model exp_4_3_StyleTransfer_train/models/fst_train.pth \
        --out ./out/my_result a.jpg b.jpg c.jpg

    # 保持长宽比（短边缩放到 512 后中心裁剪），默认是直接压成 512x512（与训练一致）
    python style_transfer_demo.py --crop photo.jpg

    # MLU 机器上跑推理（比 CPU 快很多）
    python style_transfer_demo.py --device mlu photo.jpg

输出：out/demo/<原名>_stylized.jpg，左半=原图，右半=风格化结果。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
# 直接复用课程代码里的网络定义（evaluate_cpu.py 有 __main__ 保护，import 不会触发推理）
sys.path.insert(0, os.path.join(HERE, "exp_4_2_StyleTransfer_infer", "stu_upload"))
from evaluate_cpu import TransNet  # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, "exp_4_2_StyleTransfer_infer", "models", "fst.pth")


def mlu_device(index=0):
    """返回 MLU 设备；兼容老版本(有 torch_mlu.core)和新版本(只有 torch.mlu)的 torch_mlu。"""
    import torch_mlu  # noqa: F401  确保 MLU 运行时初始化
    try:
        import torch_mlu.core.mlu_model as ct          # 老版本(1.9 ~ 1.13)接口
    except ImportError:
        ct = None                                      # 新版本(1.15+)没有这个模块
    if ct is not None and hasattr(ct, "set_device"):
        ct.set_device(index)
    elif hasattr(torch, "mlu") and hasattr(torch.mlu, "set_device"):
        torch.mlu.set_device(index)
    return torch.device("mlu", index)


def load_image(path, size=512, crop=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("读不到图片: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop:  # 短边缩放到 size，再中心裁剪为 size x size，保持长宽比
        h, w = img.shape[:2]
        scale = size / min(h, w)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        top, left = (h - size) // 2, (w - size) // 2
        img = img[top:top + size, left:left + size]
    else:     # 直接压成 size x size（与课程训练时的预处理一致）
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description="TransNet 风格迁移推理")
    ap.add_argument("images", nargs="+", help="输入图片（可多张）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="TransNet 权重 .pth")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "demo"), help="输出目录")
    ap.add_argument("--size", type=int, default=512, help="输入边长（默认 512，需能被 4 整除）")
    ap.add_argument("--crop", action="store_true", help="短边缩放+中心裁剪，保持长宽比")
    ap.add_argument("--device", choices=["cpu", "mlu", "cuda"], default="cpu", help="推理设备（MLU 机器上用 mlu）")
    args = ap.parse_args()
    if args.size % 4 != 0:
        ap.error("--size 必须是 4 的倍数（TransNet 有 2 次 stride=2 下采样 + 2 次 x2 上采样）")

    os.makedirs(args.out, exist_ok=True)
    device = mlu_device() if args.device == "mlu" else torch.device(args.device)
    net = TransNet().cpu()
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net = net.to(device).eval()
    print("model loaded: %s (device: %s)" % (args.model, device))

    with torch.no_grad():
        for path in args.images:
            image_c = load_image(path, args.size, args.crop)
            image_g = net(image_c.to(device)).cpu()
            image_c = image_c.cpu()
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.out, "%s_stylized.jpg" % name)
            save_image(torch.cat((image_c, image_g), dim=3), out_path)
            print("saved: %s" % out_path)


if __name__ == "__main__":
    main()
