# PointLoc
import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLoc, PointLocLoss
from data.dataloader import vReLocDataset
def main(*args, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    point_cloud = torch.rand(4, 3, 21000) # batch_size x 3 x N
    model = PointLoc()
    model.to(device)
    point_cloud = point_cloud.to(device)
    output = model(point_cloud)
    print(f"final output shape : {output.shape}")

    optimizer = torch.optim.Adam(model.parameters())
    schedular = torch.optim.lr_scheduler.LRScheduler
    criterion = PointLocLoss()
    num_epochs = 100

    train_dataset = vReLocDataset('', train=True)
    valid_dataset = vReLocDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    for epoch in range(num_epochs):
        training_one_epoch(model=model, optimizer=optimizer, criterion=criterion, schedular=schedular,
                           dataloader=train_loader)


def training_one_epoch(*args, **kwargs):
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    criterion = kwargs["criterion"]
    data_loader = kwargs["dataloader"]

    model.train()

    for _ in enumerate(data_loader):

        pass


def validation_one_epoch(*args, **kwargs):
    pass

if __name__ == '__main__':
    main()