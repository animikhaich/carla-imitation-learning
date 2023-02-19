import time
import os
import random
import argparse

import torch

from network import ClassificationNetwork
from dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import datetime

import time
from tqdm import tqdm

# TODO: Add Validation Dataloader and Validation Set
# DONE: Early Stopping
# TODO: Tensboard Dir in args
# TODO: If dir does not exist, create it
# TODO: Verify Model Saving Path and Model Loading Path

def train(data_folder, labels_path, save_path, use_gpus=[0], load_model=None):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Choose Training Medium
    if device == 'cuda' and len(args.use_gpus) > 1:
        infer_action = torch.nn.DataParallel(ClassificationNetwork(), device_ids=args.use_gpus).to(device)
        print("Using multiple GPUs: ", args.use_gpus)
    elif device == 'cuda' and len(args.use_gpus) == 1:
        infer_action = ClassificationNetwork().to(device)
        print("Using single GPU: ", args.use_gpus[0])
    else:
        infer_action = ClassificationNetwork().to(device)
        print("Using CPU")
    
    # Define optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e8, verbose=True)
    
    nr_epochs = 500
    best_loss = 10e10
    batch_size = 512
    start_time = time.time()

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model_name = os.path.basename(save_path).split(".")[0]
    writer = SummaryWriter(log_dir=os.path.join(args.tb_dir, f'{model_name}_{timestamp}'))
    early_stopping = EarlyStopping(patience=10, mode='lowest') # Early Stopping if loss does not increase
    
    # Load Weights if specified
    if load_model is not None and os.path.exists(load_model):
        infer_action.load_state_dict(torch.load(load_model).state_dict())
        print(f"Loaded Weights From: {load_model}")

    train_loader = get_dataloader(args.data_folder, args.labels_path, args.batch_size, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), disable=args.disable_tqdm):
            batch_in, batch_gt = batch[0].to(device), batch[1].to(device)
            batch_out = infer_action(batch_in)
            loss = torch.nn.functional.mse_loss(batch_out, batch_gt)
            
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (args.epochs - 1 - epoch)
        time_left = seconds_to_hms(time_left)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA:" % (
            epoch + 1, total_loss), time_left)

        writer.add_scalar('Loss/train', total_loss, epoch)

        best_loss = save_model(model_name, infer_action, total_loss, best_loss, mode="lowest") # Save Best and Last Model
        if early_stopping(total_loss, best_loss, mode='lowest'): # Early Stopping if loss does not increase
            print("Early Stopping")
            break

class EarlyStopping:
    def __init__(self, patience=10, mode='lowest'):
        self.patience = patience
        self.counter = 0

    def __call__(self, val_metric, best_metric, mode='lowest'):
        self.counter += 1
        if (mode == 'lowest' and val_metric <= best_metric and self.counter > self.patience) or \
            (mode == 'highest' and val_metric >= best_metric and self.counter > self.patience):
            print("Early Stopping after %d epochs" % self.counter)
            return True

def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def save_model(name, model, val_metric, best_metric, mode='lowest'):
    torch.save(model, f'{name}_last.pt')
    if mode == 'lowest':
        if val_metric <= best_metric:
            print(f"New {val_metric} <= Old {best_metric}. Saving: {f'{name}_best.pt'}")
            torch.save(model, f'{name}_best.pt')
            return val_metric
    elif mode == 'highest':
        if val_metric >= best_metric:
            print(f"New {val_metric} >= Old {best_metric}. Saving: {f'{name}_best.pt'}")
            torch.save(model, f'{name}_best.pt')
            return val_metric
    return best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="../data/rgb", type=str, help='path to where you save the images you collect')
    parser.add_argument('-l', '--labels_path', default="../data/metrics/output.csv", type=str, help='path to where you save the labels you collect')
    parser.add_argument('-m', '--model_path', default="../models/", type=str, help='path where to save your model in .pt format')
    parser.add_argument('-p', '--load_model_path', default=None, help='path to load your model in .pt format')
    args = parser.parse_args()
    
    train(args.data_folder, args.labels_path, args.model_path, load_model=args.load_model_path)
