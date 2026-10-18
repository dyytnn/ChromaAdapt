import os

import numpy as np
from dist_utils import is_main_process, save_on_master


class EarlyStopping:
    def __init__(self, config: dict):
        self.config = config
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, save_dict, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if is_main_process():
                save_on_master(save_dict, os.path.join(save_path, "last_save_dict.pth"))
                self.val_loss_min = min(self.val_loss_min, val_loss)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.config.get("patience")}')
            if self.counter >= self.config.get("patience"):
                self.early_stop = True
        else:
            self.best_score = score
            if is_main_process():
                save_on_master(save_dict, os.path.join(save_path, "last_save_dict.pth"))
                self.counter = 0
                self.val_loss_min = min(self.val_loss_min, val_loss)
                
# TODO: need to check