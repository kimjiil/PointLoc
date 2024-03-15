import torch
import torch.nn as nn
from .pointnet2_utils import PointNetSetAbstraction
import torch.nn.functional as F

class PointLoc(nn.Module):
    def __init__(self, beta0=1.0, gamma0=1.0):
        super(PointLoc, self).__init__()

    def forward(self,)

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 3)
        )

        self.rotation_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
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
        self.SA1 = PointNetSetAbstraction(npoint=2048, radius=0.2, nsample=64,in_channel=3,
                                          mlp=[64, 64, 128], group_all=False)
        self.SA2 = PointNetSetAbstraction(npoint=1024, radius=0.4, nsample=32,in_channel=128+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.SA3 = PointNetSetAbstraction(npoint=512, radius=0.8, nsample=16, in_channel=256+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.SA4 = PointNetSetAbstraction(npoint=256, radius=1.2, nsample=16, in_channel=256+3,
                                          mlp=[128, 128, 256], group_all=False)
    def forward(self, input):
        xyz, f = self.SA1(input, None)
        xyz, f = self.SA2(xyz, f)
        xyz, f = self.SA3(xyz, f)
        xyz, f = self.SA4(xyz, f)
        return xyz, f


