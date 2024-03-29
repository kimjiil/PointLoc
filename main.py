# PointLoc
import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLoc, PointLocLoss
from data.dataloader import vReLocDataset
from data.transforms import get_train_transforms, get_valid_transforms
from torch.autograd import profiler

from utils.quaternions import qexp, quaternion_angular_error
from utils.tools import Options, Ploting, DualOutput, pose_ploting

import numpy as np
from itertools import chain
import os, sys
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt

current_time = datetime.now().strftime("%Y_%m%d_%H%M_%S")
os.makedirs("./results", exist_ok=True)
os.makedirs(f"./results/{current_time}", exist_ok=True)
save_dir = f"./results/{current_time}"

file_name = os.path.join(save_dir, f"logs_{current_time}.txt")
sys.stdout = DualOutput(file_name)

def main(*args, **kwargs):
    opt = Options().parse_args()

    plot = Ploting()
    # temp = torch.load('/home/jikim/workspace/localization_ws/PointLoc/results/2024_0321_1916_36/model_E0050.pt')
    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"

    model = PointLoc()
    criterion = PointLocLoss(beta0=0.0, gamma0=-3.0)

    params = chain(model.parameters(), criterion.parameters())

    optimizer = torch.optim.Adam(params, lr=opt.lr)
    if opt.scheduler == 'CALR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50)
    elif opt.scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.97)
    else:
        scheduler = None

    train_transforms = get_train_transforms()
    valid_transforms = get_valid_transforms()

    data_path = os.path.join(opt.data_dir, opt.dataset)
    train_dataset = vReLocDataset(data_path, train=True, transform=train_transforms)
    valid_dataset = vReLocDataset(data_path, train=False, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    criterion.to(device)
    model.to(device)

    best_valid_loss = 1000
    best_valid_rotate = 360
    best_valid_trans = 20
    best_epoch = 0

    for epoch in range(opt.epochs):
        checkpoint_dict = defaultdict()

        training_one_epoch(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                           dataloader=train_loader, device=device, epoch=epoch, opt=opt, checkpoint_dict=checkpoint_dict)

        validation_one_epoch(model=model, epoch=epoch, opt=opt, dataloader=valid_loader, device=device,
                             criterion=criterion, checkpoint_dict=checkpoint_dict)


        if (checkpoint_dict['valid_rotation_error'] < best_valid_rotate and
                checkpoint_dict['valid_translation_error'] < best_valid_trans):
            print("best model saved")
            best_valid_loss = checkpoint_dict['valid_loss']
            best_valid_rotate = checkpoint_dict['valid_rotation_error']
            best_valid_trans = checkpoint_dict['valid_translation_error']
            best_epoch = checkpoint_dict['epoch']

            model_save_name = os.path.join(save_dir, f"best_model.pt")
            torch.save(checkpoint_dict, model_save_name)

        plot.plot_save(checkpoint_dict=checkpoint_dict,
                       best_epoch=best_epoch,
                       best_valid_trans=best_valid_trans,
                       best_valid_rotate=best_valid_rotate,
                       best_valid_loss=best_valid_loss,
                       save_dir=save_dir)


def training_one_epoch(*args, **kwargs):
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    criterion = kwargs["criterion"]
    data_loader = kwargs["dataloader"]
    device = kwargs["device"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    opt = kwargs["opt"]
    checkpoint_dict = kwargs["checkpoint_dict"]

    print(f"----------------- epoch {epoch} training start --------------------")

    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error

    model.train()
    criterion.train()

    pred_poses = []
    target_poses = []
    loss_list = []

    pred_pose_list = []
    gt_pose_list = []

    for batch_idx, (frame, gt_pose) in enumerate(data_loader):
        frame = frame.to(device)
        gt_pose = gt_pose.to(device)
        optimizer.zero_grad()

        if opt.memory_profile:
            # memory profiling code
            with profiler.profile(profile_memory=True, use_device=True) as prof:
                with profiler.record_function("model_inference"):
                    output = model(frame)

            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        else:
            output = model(frame)


        loss = criterion(t_pred=output[:, :3], t_gt=gt_pose[:, :3],
                         q_pred=output[:, 3:], q_gt=gt_pose[:, 3:])

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print(f"epoch {epoch} [{batch_idx}/{len(data_loader)}] training batch loss : ", loss.item())
        print(criterion)

        output = output.cpu().detach().numpy()
        q = [qexp(p[3:]) for p in output]
        pred_poses.extend(np.hstack((output[:, :3], np.asarray(q))))

        gt_pose = gt_pose.cpu().detach().numpy()
        q = [qexp(p[3:]) for p in gt_pose]
        target_poses.extend(np.hstack((gt_pose[:, :3], np.asarray(q))))

        np_target_poses = np.asarray(target_poses)
        np_pred_poses = np.asarray(pred_poses)
        t_loss = np.asarray([t_criterion(p, t) for p, t in zip(np_pred_poses[:, :3], np_target_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t) for p, t in zip(np_pred_poses[:, 3:], np_target_poses[:, 3:])])

        print('Error in translation: mean {:3.2f} m \nError in rotation: mean {:3.2f} degree' \
            .format(np.mean(t_loss), np.mean(q_loss)))
        print('============================================================')

        gt_pose_list.extend(gt_pose[:, :3])
        pred_pose_list.extend(output[:, :3])

    pose_ploting(pred_pose_list, gt_pose_list, epoch, save_dir, mode="train")
    print("*****************************************************************")
    print(f"Epoch {epoch} Total Training Loss : {np.mean(loss_list)}")
    target_poses = np.asarray(target_poses)
    pred_poses = np.asarray(pred_poses)
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], target_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], target_poses[:, 3:])])
    print('Total Error in translation: mean {:3.2f} m \nTotal Error in rotation: mean {:3.2f} degree' \
          .format(np.mean(t_loss), np.mean(q_loss)))
    print("*****************************************************************")
    if scheduler:
        scheduler.step()

    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model'] = model.state_dict()
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    checkpoint_dict['criterion'] = criterion.state_dict()
    if scheduler:
        checkpoint_dict['scheduler'] = scheduler.state_dict()
    checkpoint_dict['train_loss'] = np.mean(loss_list)
    checkpoint_dict['train_translation_error'] = np.mean(t_loss)
    checkpoint_dict['train_rotation_error'] = np.mean(q_loss)

def validation_one_epoch(*args, **kwargs):
    model = kwargs['model']
    epoch = kwargs['epoch']
    opt = kwargs['opt']
    dataloader = kwargs['dataloader']
    device = kwargs['device']
    criterion = kwargs['criterion']
    checkpoint_dict = kwargs['checkpoint_dict']

    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error

    model.eval()
    criterion.eval()

    pred_poses = []
    target_poses = []
    loss_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(t_pred=output[:, :3], t_gt=target[:, :3],
                             q_pred=output[:, 3:], q_gt=target[:, 3:])

            loss_list.append(loss.item())

            output = output.cpu().detach().numpy()
            q = [qexp(p[3:]) for p in output]
            pred_poses.extend(np.hstack((output[:, :3], np.asarray(q))))

            target = target.cpu().detach().numpy()
            q = [qexp(p[3:]) for p in target]
            target_poses.extend(np.hstack((target[:, :3], np.asarray(q))))

    pred_poses = np.asarray(pred_poses)
    target_poses = np.asarray(target_poses)

    pose_ploting(pred_poses[:, :3], target_poses[:, :3], epoch, save_dir, mode="valid")
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], target_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], target_poses[:, 3:])])

    print(f'------------------------- validation Epoch {epoch} Result ----------------------------------')
    print('Error in translation: mean {:3.2f} m \nError in rotation: mean {:3.2f} degree' \
          .format(np.mean(t_loss), np.mean(q_loss)))


    checkpoint_dict['valid_translation_error'] = np.mean(t_loss)
    checkpoint_dict['valid_rotation_error'] = np.mean(q_loss)
    checkpoint_dict['valid_loss'] = np.mean(loss_list)




if __name__ == '__main__':
    main()