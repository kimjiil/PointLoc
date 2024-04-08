import numpy as np

import matplotlib.pyplot as plt
import os, glob
from PIL import Image
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

data_path = "E:/git_repo/PointLoc/dataset/syswin"

seq_list = os.listdir(data_path)

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

    return coordinates, global_pose

current_index = 0
def update(val):
    global current_index
    current_index = int(val)
    coord, global_pose = parsing_scan_data(scan_list[current_index])

    ax1.clear()
    if coord:
        coord = np.array(coord)
        ax1.scatter(coord[:, 0], coord[:, 1], s=1)
    ax1.set_title("Scan Data")

    img = Image.open(img_list[current_index])
    ax2.imshow(img)
    ax2.set_title("Image")

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

for seq in seq_list:
    scan_list = glob.glob(os.path.join(data_path, seq, "*scan.txt"))
    img_list = glob.glob(os.path.join(data_path, seq, "*color.png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, "Index", 0, len(scan_list) - 1, valinit=current_index, valfmt='%0.0f')

    ax_button = plt.axes([0.85, 0.02, 0.1, 0.04])
    button = Button(ax_button, "Pause", color='lightgoldenrodyellow', hovercolor='0.975')

    slider.on_changed(update)
    button.on_clicked(toggle_animation)


    ani = FuncAnimation(fig, animate, interval=1)
    plt.show()


