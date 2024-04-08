import numpy as np
import matplotlib.pyplot as plt
import os, glob
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import cv2

import torch
from torch.utils.data import DataLoader
from models.PointLoc import PointLoc
from data.dataloader import SyswinTestDataset
from data.transforms import get_valid_transforms_syswin

from utils.quaternions import qexp

data_path = "E:/git_repo/PointLoc/dataset/syswin/2024_0403_1609_44"

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
model = PointLoc()

state_dict = torch.load("C:/Users/jikim/Desktop/working_temp/2024_0405_1705_14/best_model.pt", map_location=device)
model.load_state_dict(state_dict['model'])

valid_transforms = get_valid_transforms_syswin()
valid_dataset = SyswinTestDataset(data_path, transform=valid_transforms)

model.eval()
model.to(device)

def parsing_scan_data(scan_path):
    with open(scan_path, 'r') as f:
        lines = f.readlines()
        angle_min = float(lines[1].split(" ")[-1].replace("\n", ""))
        angle_max = float(lines[2].split(" ")[-1].replace("\n", ""))
        angle_step = float(lines[3].split(" ")[-1].replace("\n", ""))
        global_pose = [float(s.replace(",", "").replace("\n", "")) for s in lines[6].split(" ")[-3:]]
        distance = [float(s.replace(",", "").replace(" ", "")) for s in lines[8].split(" ") if len(s) > 1]

        angles_arr = [angle_min + angle_step * i for i in range(len(distance))]
        coordinates = [[d * np.cos(rad), d * np.sin(rad)] for rad, d in zip(angles_arr, distance) if not np.isinf(d)]
        global_coord = [[d * np.cos(rad + np.radians(global_pose[2])) + global_pose[0],
                         d * np.sin(rad + np.radians(global_pose[2])) + global_pose[1]] for rad, d in zip(angles_arr, distance) if not np.isinf(d)]
    return coordinates, global_pose, global_coord

current_index = 0
def update(val):
    global current_index
    current_index = int(val)

    point_cloud, gt_pose, img = valid_dataset[current_index]
    point_cloud = point_cloud.unsqueeze(0).contiguous()
    point_cloud = point_cloud.to(device)
    pred_pose = model(point_cloud)

    gt_pose_np = gt_pose
    pred_pose_np = pred_pose.detach().cpu().numpy()[0]

    gt_angle = np.rad2deg(np.arcsin(qexp(gt_pose_np[3:])[-1]) * 2)
    # theta = np.rad2deg(np.arctan2( qexp(gt_pose_np[3:])[-1], qexp(gt_pose_np[3:])[0]  ))
    pred_angle = np.rad2deg(np.arcsin(qexp(pred_pose_np[3:])[-1]) * 2)

    gt_translation = gt_pose_np[:3]
    pred_translation = pred_pose_np[:3]

    coord_np = point_cloud.detach().cpu().numpy()[0][:2, :]
    rot_mat_gt = np.array([[np.cos(np.radians(gt_angle)), -np.sin(np.radians(gt_angle))],
                        [np.sin(np.radians(gt_angle)), np.cos(np.radians(gt_angle))]])

    rot_mat_pred = np.array([[np.cos(np.radians(pred_angle)), -np.sin(np.radians(pred_angle))],
                        [np.sin(np.radians(pred_angle)), np.cos(np.radians(pred_angle))]])

    coord_np = np.dot(rot_mat_gt, coord_np).T

    ax1.clear()
    ax1.set_xlim(-5, 15)
    ax1.set_ylim(-8, 12)

    amr_default = np.array([[-0.7, -0.7, 0.7, 0.7, -0.7], [-.35, .35, .35, -.35, -.35]])
    gt_amr_box = np.dot(rot_mat_gt, amr_default ).T
    pred_amr_box = np.dot(rot_mat_pred, amr_default).T
    ax1.plot(gt_amr_box[:, 0] + gt_translation[0], gt_amr_box[:, 1] + gt_translation[1], color='blue')
    ax1.plot(pred_amr_box[:, 0] + pred_translation[0], pred_amr_box[:, 1] + pred_translation[1], color='red')
    ax1.scatter(coord_np[:, 0] + gt_translation[0], coord_np[:, 1] + gt_translation[1], s=0.5, color='black')
    ax1.plot(gt_translation[0], gt_translation[1], '*', markersize=5, color='blue')
    ax1.plot(pred_translation[0], pred_translation[1], '*', markersize=5, color='red')
    ax1.set_title("Scan Data")

    cv2.imshow("Scan Data", img)

    fig.canvas.draw_idle()

animation_running = True

def toggle_animation(butten_press):
    global animation_running
    if animation_running:
        ani.event_source.stop()
        button.label.set_text("Play")
    else:
        ani.event_source.start()
        button.label.set_text("Pause")
    animation_running = not animation_running

def animate(frame):
    slider.set_val((slider.val + 1) % len(scan_list))


scan_list = glob.glob(os.path.join(data_path, "*scan.txt"))
img_list = glob.glob(os.path.join(data_path, "*color.png"))

loaded_coord = []
loaded_global_pose = []
loaded_global_coord = []
for scan_path in scan_list:
    coordinates, global_pose, global_coordinates = parsing_scan_data(scan_path)
    loaded_coord.append(coordinates)
    loaded_global_pose.append(global_pose)
    loaded_global_coord.append(global_coordinates)

fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.13, 0.01, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax_slider, "Index", 0, len(scan_list) - 1, valinit=current_index, valfmt='%0.0f')

ax_button = plt.axes([0.88, 0.01, 0.1, 0.04])
button = Button(ax_button, "Pause", color='lightgoldenrodyellow', hovercolor='0.975')

slider.on_changed(update)
button.on_clicked(toggle_animation)

ani = FuncAnimation(fig, animate, interval=1)

plt.show()
cv2.destroyAllWindows()


