import os
import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
import glob
from utils.quaternions import process_poses



class vReLocDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, "full")
        self.transform = transform

        txt_file = "TrainSplit.txt" if train else "TestSplit.txt"

        seqs = []
        with open(os.path.join(self.root, txt_file), 'r', encoding='utf-8') as txt_f:
            for line in txt_f:
                if not '#' in line:
                    seq = line.replace("\n", "")
                    seq = seq[:3] + "-" + seq[8:].rjust(2, "0")
                    seqs.append(os.path.join(self.root, seq))

        self.frames = []
        self.poses = np.empty((0, 6))
        self.size_list = []
        for seq in seqs:
            ps = []
            bin_list = sorted(glob.glob(os.path.join(seq, "*.bin")), key=lambda x: int(x.split("frame")[-1].replace("-", "").replace(".bin", "")))
            for bin_file_path in bin_list:
                lidar_file = np.fromfile(bin_file_path, dtype=np.float32).reshape((4, -1))[:3, :]
                self.size_list.append(lidar_file.shape[-1])
                pose = np.loadtxt(bin_file_path.replace(".bin", ".pose.txt"), delimiter=",") # homogenous coordinate
                pose = pose.flatten()[:12]
                ps.append(pose)
                self.frames.append(lidar_file)

            ps = np.array(ps)
            pss = process_poses(ps)
            self.poses = np.vstack((self.poses, pss))



    def __len__(self):
        return len(self.poses)
        # return 128

    def __getitem__(self, idx):
        frame = self.frames[idx]
        pose = self.poses[idx]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, pose

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)



if __name__ == '__main__':
    path = "E:/DeepLearning/localization/자료/(2022) PointLoc/vReLoc/full/seq-01/frame-000000.bin"

    vReLocDataset("E:/DeepLearning/localization/자료/(2022) PointLoc/vReLoc")

    data = np.fromfile(path, dtype=np.float32)
    ptcld = data.reshape((4, -1)).T


    import torch
    import numpy as np

    # 가정된 변환 행렬 (사용자의 pose.txt 파일로부터 얻은 실제 데이터를 여기에 넣으세요)
    transformation_matrix = torch.tensor([
        [0.999988, -0.004778, 0.000877, 0.026211],
        [0.004778, 0.999989, 0.000033, -0.011776],
        [-0.000860, -0.000097, 0.999999, 0.011784],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 이동 벡터 추출
    translation_vector = transformation_matrix[:3, 3]

    # 회전 행렬 추출
    rotation_matrix = transformation_matrix[:3, :3]

    # 쿼터니언으로 변환 (PyTorch의 경우)
    quaternion = torch.tensor(matrix_to_quaternion(rotation_matrix))

    # 결과 출력
    print("Translation Vector:\n", translation_vector.numpy())
    print("Quaternion:\n", quaternion.numpy())

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # 주어진 쿼터니언
    quaternion = np.array([0.99999702, -3.2500098e-05, 4.3425127e-04, 2.3890072e-03])

    # 쿼터니언을 회전 객체로 변환
    rotation = R.from_quat(quaternion)

    # 오일러 각도로 변환 (단위: 라디안)
    euler_rad = rotation.as_euler('xyz', degrees=False)

    # 오일러 각도를 도(degree) 단위로 변환
    euler_deg = np.degrees(euler_rad)
    print(euler_deg)