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

class ToTensor(object):
    def __call__(self, input):
        return torch.from_numpy(input)

def get_transforms():
    tf = transforms.Compose([
        RandomSampling(20000),
        ToTensor()
    ])

    return tf

if __name__ == '__main__':
    point_cloud = torch.rand(4, 3, 100)  # batch_size x 3 x N
    rs = RandomSampling(50)
    sp = rs(point_cloud)
    print()