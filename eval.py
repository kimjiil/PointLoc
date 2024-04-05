import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLocLoss, PointLoc
from data.dataloader import vReLocDataset
from data.transforms import get_valid_transforms, get_train_transforms

import os

from utils.tools import Options, pose_ploting

import numpy as np


def rotation_matrix_to_euler_angles(R):
    """
    ZYX 회전 순서를 사용하여 회전 행렬 R에서 오일러 각을 계산합니다.

    인자:
    - R: 3x3 회전 행렬

    반환값:
    - [yaw, pitch, roll]: 오일러 각(라디안 단위)
    """
    # 안전성 체크: r_31 값이 -1과 1 사이에 있는지 확인 (짐벌락 방지)
    if R[2, 0] != -1 and R[2, 0] != 1:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # r_31 = -1의 경우 (pitch = +90도)
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        # r_31 = 1의 경우 (pitch = -90도)
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
        yaw = 0  # 짐벌락 상태에서는 요와 롤의 구분이 불가능

    return np.array([yaw, pitch, roll])
import matplotlib.pyplot as plt
import matplotlib
def main(*args, **kwargs):
    opt = Options().parse_args()

    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    model = PointLoc()

    # st = torch.load("/home/jikim/workspace/localization_ws/PointLoc/results/2024_0326_1646_54/best_model.pt", map_location=device)
    # st = torch.load("/home/jikim/workspace/localization_ws/PointLoc/results/2024_0325_1050_20/model_E0520.pt", map_location=device)
    st = torch.load("C:/Users/jikim/Desktop/temp/2024_0328_1846_10/best_model.pt", map_location=device)

    model.load_state_dict(st['model'])

    valid_transforms = get_valid_transforms()
    data_path = os.path.join(opt.data_dir, opt.dataset)
    valid_dataset = vReLocDataset(data_path, train=False, transform=valid_transforms)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    model.to(device)
    model.eval()
    gt_pose_list = []
    pred_pose_list = []
    angle_pose_list = []
    for batch_idx, (point_cloud, gt_pose, rot_mat) in enumerate(valid_loader):
        point_cloud = point_cloud.to(device)
        pred_pose = model(point_cloud)

        gt_pose_np = gt_pose.cpu().detach().numpy()
        pred_pose_np = pred_pose.cpu().detach().numpy()

        gt_pose_list.extend(gt_pose_np)
        pred_pose_list.extend(pred_pose_np)

        # yaw(z), pitch(y), roll(x)
        euler_angle = rotation_matrix_to_euler_angles(rot_mat[0].detach().numpy())
        angle_pose_list.append(euler_angle)
        # print(euler_angle)
        # print()

    pose_ploting_test(pred_pose_list, gt_pose_list, angle_pose_list, st["epoch"],"./", "eval")
    print()

def pose_ploting_test(pred_pose, gt_pose, angle_pose_list,epoch, save_dir, mode="train"):
    pred_pose = np.asarray(pred_pose)
    gt_pose = np.asarray(gt_pose)
    angle = np.asarray(angle_pose_list)
    plt.clf()
    plt.title(f"{mode} pose plot")
    plt.quiver(gt_pose[:, 0], gt_pose[:, 1], np.cos(angle[:, 0]), np.sin(angle[:, 0]))
    plt.plot(pred_pose[:, 0], pred_pose[:, 1], color="red", label="pred_pose", linewidth=0.1)
    plt.plot(gt_pose[:, 0], gt_pose[:, 1], color='blue', label="gt_poas", linewidth=0.1)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"E{str(epoch).rjust(4, str(0))}_{mode}.png"))


if __name__ == "__main__":
    main()