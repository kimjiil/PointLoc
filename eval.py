import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLocLoss, PointLoc
from data.dataloader import vReLocDataset
from data.transforms import get_valid_transforms, get_train_transforms

import os

from utils.tools import Options

import matplotlib.pyplot as plt
import matplotlib
def main(*args, **kwargs):
    opt = Options().parse_args()

    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    model = PointLoc()

    # st = torch.load("/home/jikim/workspace/localization_ws/PointLoc/results/2024_0326_1646_54/best_model.pt", map_location=device)
    # st = torch.load("/home/jikim/workspace/localization_ws/PointLoc/results/2024_0325_1050_20/model_E0520.pt", map_location=device)
    st = torch.load("/home/jikim/workspace/localization_ws/PointLoc/results/2024_0321_1916_36/model_E1999.pt", map_location=device)
    model.load_state_dict(st['model'])

    valid_transforms = get_valid_transforms()
    data_path = os.path.join(opt.data_dir, opt.dataset)
    valid_dataset = vReLocDataset(data_path, train=False, transform=valid_transforms)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    model.to(device)
    model.eval()
    move_x = []
    move_y = []
    for batch_idx, (point_cloud, gt_pose) in enumerate(valid_loader):
        point_cloud = point_cloud.to(device)
        output = model(point_cloud)

        # point_cloud_np = point_cloud.detach().numpy()
        gt_pose_np = gt_pose.detach().numpy()
        output_np = output.detach().cpu().numpy()
        move_x.append(gt_pose_np[0][0])
        move_y.append(gt_pose_np[0][1])
        plt.plot(gt_pose_np[0][0], gt_pose_np[0][1], '*', markersize=1.5, color='blue')
        plt.plot(output_np[0][0], output_np[0][1], '*', markersize=1.5, color='red')
        if batch_idx % 10 == 0:
            print()
    plt.plot(move_x, move_y, 'o')
    print()



if __name__ == "__main__":
    main()