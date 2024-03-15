# PointLoc
import torch
from models.PointLoc import PointLoc
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
    criterion = torch.nn.MSELoss()
    num_epochs = 100
    for epoch in range(num_epochs):
        training_one_epoch(model=model, optimizer=optimizer, criterion=criterion, schedular=schedular)


def training_one_epoch(*args, **kwargs):
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    criterion = kwargs["criterion"]

    model.train()


def validation_one_epoch(*args, **kwargs):
    pass

if __name__ == '__main__':
    main()