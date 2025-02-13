import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.decode_pose import decode_pose
from utils.openpose_net import OpenPoseNet

# モデルの定義
net = OpenPoseNet()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 学習済みパラメータのロード
net_weights = torch.load("./weights/ILLUST100.pth", map_location=device)
keys = list(net_weights.keys())
weights_load = {}
for i in range(len(keys)):
	weights_load[list(net.state_dict().keys())[i]] = net_weights[list(keys)[i]]
state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

net.eval()

# 画像の読み込みと前処理
if len(sys.argv) != 2:
	print("Usage: python predict_single_image.py <image_path>")
	exit(1)

img_path = sys.argv[1]
ori_img = cv2.imread(img_path)
if ori_img is None:
	print("画像が読み込めませんでした:", img_path)
	exit(1)

# BGRからRGBに変換
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

# 画像のリサイズ
size = (368, 368)
img = cv2.resize(ori_img, size, interpolation=cv2.INTER_CUBIC)
img = img.astype(np.float32) / 255.0

# 色情報の標準化（必要に応じて調整してください）
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]
for i in range(3):
	img[:, :, i] = (img[:, :, i] - color_mean[i]) / color_std[i]

# (高さ, 幅, 色) → (色, 高さ, 幅) に並び替え
img = img.transpose((2, 0, 1))
img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

# ネットワークで推論し、予測結果を処理する
with torch.no_grad():
	predicted_outputs, _ = net(img_tensor)

# CPU上でnumpy配列に変換
pafs = predicted_outputs[0][0].cpu().numpy().transpose(1, 2, 0)
heatmaps = predicted_outputs[1][0].cpu().numpy().transpose(1, 2, 0)

# 元の画像サイズにリサイズ
original_size = (ori_img.shape[1], ori_img.shape[0])
pafs = cv2.resize(pafs, original_size, interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, original_size, interpolation=cv2.INTER_CUBIC)

# decode_poseを使用して姿勢推定結果を取得
# 戻り値を実際の関数の戻り値に合わせて修正
to_plot, canvas, joint_list, person_to_joint_assoc = decode_pose(
	ori_img, heatmaps, pafs
)

# 人数と関節の表示
print("\nDetected Keypoints:")
if len(person_to_joint_assoc) > 0:
	for person_idx, person in enumerate(person_to_joint_assoc):
		print(f"\nPerson {person_idx + 1}:")
		for joint_type in range(18):  # NUM_JOINTS = 18
			joint_index = int(person[joint_type])
			if joint_index == -1:
				continue
			kp = joint_list[joint_index]
			# kp is [x, y, score, joint_type]. If kp values are 0-dim arrays, use .item()
			x_val = kp[0] if np.isscalar(kp[0]) else kp[0].item()
			y_val = kp[1] if np.isscalar(kp[1]) else kp[1].item()
			print(f"Keypoint {joint_type}: ({x_val:.1f}, {y_val:.1f})")
else:
	print("No person detected")


# 結果の表示
mplt_fig, mplt_axes = plt.subplots(1, 3, figsize=(18, 6))

# 元の画像
mplt_axes[0].imshow(ori_img)
mplt_axes[0].set_title("Original Image")
mplt_axes[0].axis("off")

# ヒートマップ
heatmap_channel = heatmaps[:, :, 0]
mplt_axes[1].imshow(heatmap_channel)
mplt_axes[1].set_title("Heatmap for Joint 0")
mplt_axes[1].axis("off")

# 姿勢推定結果
mplt_axes[2].imshow(canvas)
mplt_axes[2].set_title("Pose Estimation Result")
mplt_axes[2].axis("off")

plt.tight_layout()
plt.show()
