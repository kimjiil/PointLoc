import argparse
import os


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