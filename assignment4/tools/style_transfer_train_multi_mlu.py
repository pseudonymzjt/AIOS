# -*- coding: utf-8 -*-
"""多风格 / 自定义内容图 的风格迁移训练脚本 —— MLU 版本。

与 tools/style_transfer_train_multi.py 完全同构，只是把 VGG19 特征网络、TransNet 生成器、
输入张量和损失函数都搬到 MLU 上；风格目标（多张风格图 Gram 的加权平均）也在 MLU 上算。
保存权重前统一搬回 CPU，这样训出来的 .pth 在 CPU / MLU 上都能直接用。

用法（远程 MLU 机器）：
    python tools/style_transfer_train_multi_mlu.py \
        --styles data/udnie.jpg data/sketch.jpg --weights 0.7 0.3 \
        --epochs 4 --out models/fst_mixed.pth

    # 自己的照片当内容图
    python tools/style_transfer_train_multi_mlu.py \
        --styles data/udnie.jpg --content-dir /path/to/photos --out models/fst_mine.pth

训练完用推理脚本验证（CPU 或 MLU 推理都支持）：
    python style_transfer_demo.py --model models/fst_mixed.pth --device mlu photo.jpg
"""
import argparse
import glob
import os
import sys

# MLU 设备编号必须在 import torch_mlu 之前设置
os.putenv('MLU_VISIBLE_DEVICES', '0')

import cv2
import torch
import torch_mlu
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# 兼容不同版本的 torch_mlu：老版本(1.9 ~ 1.13)有 torch_mlu.core，新版(1.15+)已去掉
try:
    import torch_mlu.core.mlu_model as ct
except ImportError:
    ct = None


def mlu_device(index=0):
    """返回 MLU 设备；老版本走 torch_mlu.core，新版本走 torch.mlu / torch.device('mlu')"""
    if ct is not None and hasattr(ct, 'set_device'):
        ct.set_device(index)
    elif hasattr(torch, 'mlu') and hasattr(torch.mlu, 'set_device'):
        torch.mlu.set_device(index)
    return torch.device('mlu', index)


HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(HERE, "..", "exp_4_3_StyleTransfer_train"))
sys.path.insert(0, os.path.join(EXP_DIR, "stu_upload"))
# train.py 有 __main__ 保护，import 不会触发训练
from train import TransNet, VGG19, get_gram_matrix, load_image, COCODataSet  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FolderImageDataset(Dataset):
    """用任意文件夹里的图片当内容图集（预处理与课程代码一致：512x512、RGB、[0,1]）。"""

    def __init__(self, folder):
        self.files = sorted(
            p for p in glob.glob(os.path.join(folder, "*"))
            if p.lower().endswith(IMG_EXTS)
        )
        if not self.files:
            raise SystemExit("错误: 目录里没有图片 -> %s" % folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return load_image(self.files[item])[0]   # load_image 返回 1x3x512x512


def merged_style_targets(net, style_paths, weights, device):
    """把多张风格图的 Gram 矩阵加权平均，得到 4 层风格目标（在 MLU 上计算）。"""
    per_style = []
    with torch.no_grad():
        for path in style_paths:
            x = load_image(path).to(device)
            per_style.append([get_gram_matrix(f) for f in net(x)])
    if weights is None:
        weights = [1.0] * len(style_paths)
    total = float(sum(weights))
    merged = []
    for layer in range(4):
        acc = sum(w * per_style[i][layer] for i, w in enumerate(weights)) / total
        merged.append(acc.detach())
    return merged


def cpu_state_dict(model):
    """保存前把权重搬回 CPU，保证生成的文件在 CPU/MLU 上都能加载。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser(description="多风格 / 自定义内容图 的风格迁移训练（MLU）")
    ap.add_argument("--styles", nargs="+", required=True, help="风格图（可多张）")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="每个风格的权重，默认等权")
    ap.add_argument("--content-dir", default=None, help="内容图文件夹；不给则用课程 COCO 数据")
    ap.add_argument("--epochs", type=int, default=1, help="训练轮数（每轮 = 遍历一遍内容图集）")
    ap.add_argument("--steps", type=int, default=None,
                    help="总训练步数（优先于 --epochs；内容图很少时用这个控制训练量）")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    ap.add_argument("--style-weight", type=float, default=5e-9, help="风格损失权重，默认 5e-9")
    ap.add_argument("--out", default=os.path.join(EXP_DIR, "models", "fst_multi_mlu.pth"), help="输出权重路径")
    ap.add_argument("--preview-dir", default=os.path.join(EXP_DIR, "out", "train_multi_mlu"), help="预览图目录")
    ap.add_argument("--preview-every", type=int, default=20, help="每隔多少步存一次权重和预览图")
    args = ap.parse_args()

    for p in args.styles:
        if not os.path.isfile(p):
            raise SystemExit("错误: 风格图不存在 -> %s" % p)
    if args.weights is not None and len(args.weights) != len(args.styles):
        raise SystemExit("错误: --weights 数量应与 --styles 一致")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)

    device = mlu_device()
    print("device:", device)

    net = VGG19().cpu().to(device)
    g_net = TransNet().cpu().to(device)
    optimizer = torch.optim.Adam(g_net.parameters(), lr=args.lr)
    loss_func = nn.MSELoss().cpu().to(device)

    style_targets = merged_style_targets(net, args.styles, args.weights, device)
    print("style targets ready: %d styles, weights=%s" % (len(args.styles), args.weights or "equal"))

    if args.content_dir:
        data_set = FolderImageDataset(args.content_dir)
        print("content dataset: %s (%d images)" % (args.content_dir, len(data_set)))
    else:
        data_set = COCODataSet()
        print("content dataset: ./data/train2014.zip (%d images)" % len(data_set))

    data_loader = DataLoader(data_set, batch_size=1, shuffle=True, drop_last=True)

    if args.steps:  # 用总步数控制训练量：把 epochs 换算成够用的轮数
        args.epochs = max(1, -(-args.steps // max(1, len(data_set))))
    print("epochs:", args.epochs, "steps target:", args.steps or "epochs*len(dataset)")

    step = 0
    stop = False
    for epoch in range(max(1, args.epochs)):
        if stop:
            break
        for i, image_c in enumerate(data_loader):
            image_c = image_c.to(device)
            image_g = g_net(image_c)
            outs = net(image_g)

            loss_s = sum(
                loss_func(get_gram_matrix(outs[k]), style_targets[k]) for k in range(4)
            )
            c2 = net(image_c)[1]
            loss_c = loss_func(outs[1], c2.detach())
            loss = loss_c + args.style_weight * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, "step:", step, "loss:", round(loss.detach().item(), 4),
                  "loss_c:", round(loss_c.detach().item(), 4),
                  "loss_s:%.3g" % loss_s.detach().item())
            if step % max(1, args.preview_every) == 0:
                torch.save(cpu_state_dict(g_net), args.out)
                save_image(torch.cat((image_g.detach().cpu(), image_c.cpu()), dim=3),
                           os.path.join(args.preview_dir, "%d.jpg" % step))
            step += 1
            if args.steps and step >= args.steps:
                stop = True
                break

    torch.save(cpu_state_dict(g_net), args.out)
    print("saved model:", args.out)
    print("MLU TRAIN RESULT PASS!")


if __name__ == "__main__":
    main()
