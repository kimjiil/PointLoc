import torch
import torch.nn as nn
from .pointnet2_utils import PointNetSetAbstraction as PSA1
from .flownet3d_utils import PointNetSetAbstraction as PSA2
import torch.nn.functional as F

import matplotlib.pyplot as plt

def quaternion_logarithm(q):
    """
    Calculate the logarithm of a quaternion.
    """
    q_norm = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
    q_w = q[:, 0].unsqueeze(1)
    vec_part = q[:, 1:]

    # Avoid division by zero
    small_q_norm = q_norm < 1e-12
    q_norm = torch.where(small_q_norm, torch.ones_like(q_norm), q_norm)

    log_q = torch.zeros_like(q)
    log_q[:, 0] = torch.log(torch.norm(q, p=2, dim=1))
    log_q[:, 1:] = (vec_part / q_norm) * torch.acos(q_w / torch.norm(q, p=2, dim=1, keepdim=True))

    return log_q


class PointLocLoss(nn.Module):
    def __init__(self, beta0=1.0, gamma0=1.0):
        super(PointLocLoss, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta0]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([gamma0]), requires_grad=True)

    def forward(self, t_pred, t_gt, q_pred, q_gt):
        # Translation error
        t_error = torch.norm(t_pred - t_gt, p=1, dim=1)

        # Quaternion logarithm error for rotation
        log_q_pred = quaternion_logarithm(q_pred)
        log_q_gt = quaternion_logarithm(q_gt)
        q_error = torch.norm(log_q_pred - log_q_gt, p=1, dim=1)

        # PointLoc loss calculation
        loss = torch.mean(t_error * torch.exp(-self.beta) + self.beta + q_error * torch.exp(-self.gamma) + self.gamma)
        return loss

    def __str__(self):
        "test"
        return f"PointLoc Loss beta: {round(self.beta.item(), 7)} / gamma: {round(self.gamma.item(), 7)}"
        # print(f"PointLoc Loss beta: {round(self.beta.item(), 7)} / gamma: {round(self.gamma.item(), 7)}")

class PointLoc(nn.Module):
    def __init__(self):
        super(PointLoc, self).__init__()
        self.point_cloud_encoder = PointCloudEncoder()
        self.self_attention_module = SelfAttentionModule(feature_dim=1)
        self.group_all_layers_module = GroupAllLayersModule()
        self.pose_regressor = PoseRegressor()
    def forward(self, input):
        # input : batch_size x channel x n_points
        xyz, feature = self.point_cloud_encoder(input)
        # feature : batch_size x channel x n_points
        weighted_feature = self.self_attention_module(feature)
        # weighted_feature : batch_size x channels x n_points
        feature_vectors = self.group_all_layers_module(weighted_feature)
        # feature_vectors : batch_size x channels
        pose_reg = self.pose_regressor(feature_vectors)
        # pose_reg : batch_size x 6 [translation(3), rotation(3)]
        return pose_reg


class PoseRegressor(nn.Module):
    def __init__(self):
        super(PoseRegressor, self).__init__()

        self.translation_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(64, 3)
        )

        self.rotation_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(64, 3)
        )

    def forward(self, input):
        t = self.translation_mlp(input)
        r = self.rotation_mlp(input)
        return torch.cat([t, r], dim=-1)
class GroupAllLayersModule(nn.Module):
    def __init__(self):
        super(GroupAllLayersModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02, inplace=True),
            # nn.Dropout(0.4)
        )
        self.fc_layer = nn.Linear(in_features=1024, out_features=1024)


    def forward(self, input):
        for l in self.mlp:
            input = l(input)
        # batch_size x channels x n_points
        input = F.max_pool1d(input, input.size(-1)).squeeze(-1)
        input = self.fc_layer(input)

        return input

class SelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttentionModule, self).__init__()

        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    def forward(self, feature):
        # feature : batch_size x Channel x n_points
        mask = self.mlp(feature)
        # broadcast mask to match feature
        weighted_feature = feature * mask
        return weighted_feature

class PointCloudEncoder(nn.Module):
    def __init__(self):
        super(PointCloudEncoder, self).__init__()
        self.SA1 = PSA2(npoint=2048, radius=0.2, nsample=64,in_channel=3,
                                          mlp=[64, 64, 128], group_all=False)
        self.SA2 = PSA2(npoint=1024, radius=0.4, nsample=32,in_channel=128+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.SA3 = PSA2(npoint=512, radius=0.8, nsample=16, in_channel=256+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.SA4 = PSA2(npoint=256, radius=1.2, nsample=16, in_channel=256+3,
                                          mlp=[128, 128, 256], group_all=False)
    def forward(self, input):

        # input_t = input.detach().cpu().numpy()[0]
        # plt.plot(input_t[0, :], input_t[1, :], 'o', markersize=1, color='green')
        xyz, f = self.SA1(input, None)
        # xyz_t = xyz.detach().cpu().numpy()[0].reshape(3, -1)
        # plt.plot(xyz_t[0, :], xyz_t[1, :], 'o', markersize=1, color='red')
        xyz, f = self.SA2(xyz, f)
        # xyz_t = xyz.detach().cpu().numpy()[0].reshape(3, -1)
        # plt.plot(xyz_t[0, :], xyz_t[1, :], 'o', markersize=1, color='blue')
        xyz, f = self.SA3(xyz, f)
        xyz, f = self.SA4(xyz, f)
        return xyz, f


