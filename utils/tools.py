import argparse
import os
import matplotlib.pyplot as plt
import sys

import numpy as np


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--data_dir", type=str, default="./dataset")
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--epochs", type=int, default=200)
        self.parser.add_argument("--lr", type=float, default=0.001)
        self.parser.add_argument("--dataset", type=str, default="vReLoc")
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--gpu_id", type=int, default=0)
        self.parser.add_argument("--memory_profile", type=bool, default=False)
        self.parser.add_argument('--scheduler', type=str, default=None)
        self.opt = self.parser.parse_args()

    def parse_args(self):
        return self.opt


class Ploting:
    def __init__(self):
        self.epochs = []
        self.valid_loss = []
        self.train_loss = []
        self.train_translation_error = []
        self.valid_translation_error = []
        self.train_rotation_error = []
        self.valid_rotation_error = []

    def plot_save(self, *args, **kwargs):
        checkpoint_dict = kwargs['checkpoint_dict']
        save_dir = kwargs['save_dir']
        self.epochs.append(checkpoint_dict['epoch'])
        self.valid_loss.append(checkpoint_dict['valid_loss'])
        self.train_loss.append(checkpoint_dict['train_loss'])
        self.train_translation_error.append(checkpoint_dict['train_translation_error'])
        self.valid_translation_error.append(checkpoint_dict['valid_translation_error'])
        self.train_rotation_error.append(checkpoint_dict['train_rotation_error'])
        self.valid_rotation_error.append(checkpoint_dict['valid_rotation_error'])

        best_epoch = kwargs['best_epoch']
        best_valid_loss = kwargs['best_valid_loss']
        best_valid_rotate = kwargs['best_valid_rotate']
        best_valid_trans = kwargs['best_valid_trans']
        plt.clf()
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.text(best_epoch, best_valid_loss, f"best E{best_epoch} - {round(best_valid_loss, 6)}")
        plt.plot(self.epochs, self.train_loss, label='Train Loss', marker='o')
        plt.plot(self.epochs, self.valid_loss, label='Valid Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 이동 오류(Translation Error) 시각화
        plt.subplot(1, 3, 2)
        plt.text(best_epoch, best_valid_trans, f"best E{best_epoch} - {round(best_valid_trans, 5)}")
        plt.plot(self.epochs, self.train_translation_error, label='Train Translation Error', marker='o')
        plt.plot(self.epochs, self.valid_translation_error, label='Valid Translation Error', marker='o')
        plt.title('Training and Validation Translation Error')
        plt.xlabel('Epoch')
        plt.ylabel('Translation Error (units)')
        plt.legend()

        # 회전 오류(Rotation Error) 시각화
        plt.subplot(1, 3, 3)
        plt.text(best_epoch, best_valid_rotate, f"best E{best_epoch} - {round(best_valid_rotate)}")
        plt.plot(self.epochs, self.train_rotation_error, label='Train Rotation Error', marker='o')
        plt.plot(self.epochs, self.valid_rotation_error, label='Valid Rotation Error', marker='o')
        plt.title('Training and Validation Rotation Error')
        plt.xlabel('Epoch')
        plt.ylabel('Rotation Error (degrees)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot.png"))

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self): # 필요한 경우, 출력 버퍼를 flush하는 데 사용
        # 이 메소드는 print 함수가 파일과 터미널에 출력한 후 버퍼를 비우는 데 필요합니다.
        self.terminal.flush()
        self.log.flush()

def pose_ploting(pred_pose, gt_pose, epoch, save_dir, mode="train"):
    pred_pose = np.asarray(pred_pose)
    gt_pose = np.asarray(gt_pose)
    plt.clf()
    plt.title(f"{mode} pose plot")
    plt.plot(pred_pose[:, 0], pred_pose[:, 1], 'o', markersize=1, color="red", label="pred_pose")
    plt.plot(gt_pose[:, 0], gt_pose[:, 1], 'o', markersize=1, color='blue', label="gt_poas")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"E{str(epoch).rjust(4, str(0))}_{mode}.png"))
