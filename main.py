# PointLoc
import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLoc, PointLocLoss
from data.dataloader import vReLocDataset
from data.transforms import get_transforms
from torch.autograd import profiler

def main(*args, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model input example
    # point_cloud = torch.rand(4, 3, 21000) # batch_size x 3 x N
    # model = PointLoc()
    # model.to(device)
    # point_cloud = point_cloud.to(device)
    # output = model(point_cloud)
    # print(f"final output shape : {output.shape}")

    model = PointLoc()
    optimizer = torch.optim.Adam(model.parameters())
    schedular = torch.optim.lr_scheduler.LRScheduler
    criterion = PointLocLoss()
    num_epochs = 100
    train_transforms = get_transforms()
    train_dataset = vReLocDataset('./dataset/vReLoc', train=True, transform=train_transforms)
    # valid_dataset = vReLocDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion.to(device)
    model.to(device)

    for epoch in range(num_epochs):
        training_one_epoch(model=model, optimizer=optimizer, criterion=criterion, schedular=schedular,
                           dataloader=train_loader, device=device)


def training_one_epoch(*args, **kwargs):
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    criterion = kwargs["criterion"]
    data_loader = kwargs["dataloader"]
    device = kwargs["device"]

    model.train()
    for batch_idx, (frame, gt_pose) in enumerate(data_loader):
        frame = frame.to(device)
        gt_pose = gt_pose.to(device)
        optimizer.zero_grad()
        # memory profiling code
        # with profiler.profile(profile_memory=True, use_device=True) as prof:
        #     with profiler.record_function("model_inference"):
        #         output = model(frame)
        #
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

        output = model(frame)
        loss = criterion(t_pred=output[:, :3], t_gt=gt_pose[:, :3],
                         q_pred=output[:, 3:], q_gt=gt_pose[:, 3:])

        loss.backward()
        optimizer.step()
        print(batch_idx, loss.item())


def validation_one_epoch(*args, **kwargs):
    pass

if __name__ == '__main__':
    main()