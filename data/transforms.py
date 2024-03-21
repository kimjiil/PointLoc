import torch
from torchvision import transforms, utils
import numpy as np

class RandomSampling(object):
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, input):
        indices = np.random.permutation(input.shape[1])[:self.n_sample]
        sampled_points = input[:, indices]

        return sampled_points

class Randomjitter(object):
    """
        포인트 클라우드에 랜덤 지터링을 추가하는 함수.

        파라미터:
        - point_cloud: (N, 3) 형태의 NumPy 배열. N은 포인트의 수, 각 행은 XYZ 좌표.
        - sigma: 지터링의 표준편차.
        - clip: 지터링 값의 최대 절대값. 이 값을 넘는 지터링은 이 값으로 클립됩니다.

        반환:
        - 지터링이 추가된 포인트 클라우드.
        """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, point_cloud):
        _, N = point_cloud.shape
        jitter = np.clip(self.sigma * np.random.randn(3, N), -self.clip, self.clip)
        point_cloud_jittered = point_cloud + jitter
        return point_cloud_jittered

class RandomRotation(object):
    "z축을 기준으로 회전"
    def __init__(self):
        pass

    def __call__(self, point_cloud, theta=None):
        if theta is None: #test 용
            theta = np.random.uniform(0, np.pi * 2)
            
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        rotated_point_cloud = np.dot(rotation_matrix, point_cloud)
        return rotated_point_cloud

class RandomTranslation(object):
    def __init__(self, translation_range=0.2):
        self.translation_range = translation_range

    def __call__(self, point_cloud):
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=(3, ))
        translated_point_cloud = point_cloud + translation

        return translated_point_cloud

class RandomSamplingRatio(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, point_cloud):
        num_points_to_select = int(point_cloud.shape[1] * self.ratio)
        selected_indices = np.random.choice(point_cloud.shape[1], num_points_to_select, replace=False)
        selected_point_cloud = point_cloud[:, selected_indices]

        return selected_point_cloud

class ToTensor(object):
    def __call__(self, input):
        return torch.from_numpy(input.astype(np.float32))

def get_train_transforms():
    tf = transforms.Compose([
        RandomSampling(20000),
        Randomjitter(sigma=0.01, clip=0.05),
        RandomSamplingRatio(ratio=0.9),
        ToTensor()
    ])

    return tf

def get_valid_transforms():
    tf = transforms.Compose([
        # RandomSampling(20000),
        ToTensor()
    ])

    return tf

if __name__ == '__main__':
    point_cloud = torch.rand(4, 3, 100)  # batch_size x 3 x N
    rs = RandomSampling(50)
    sp = rs(point_cloud)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    point_cloud = np.array([
        [1.0, 0.5, 0.3],
        [0.4, 0.2, 0.1],
        [0.6, 0.8, 0.9]
    ])


    point_cloud = np.array([
        [1.0],
        [0.0],
        [0.0]
    ])

    rot = RandomRotation()
    rot_point = rot(point_cloud, theta = np.pi / 2)

    print(rot_point)
    print()