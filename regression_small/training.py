import time
import os
import random
import argparse

import torch

from network import ClassificationNetwork
from dataset import CarlaDataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

import time
from tqdm import tqdm

# TODO: Add Validation Dataloader and Validation Set
# DONE: Early Stopping
# DONE: Tensboard Dir in args
# DONE: Fix Args
# DONE: If dir does not exist, create it
# DONE: Verify Model Saving Path and Model Loading Path

def train(args):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Choose Training Medium
    if device == 'cuda' and len(args.use_gpus) > 1:
        infer_action = torch.nn.DataParallel(ClassificationNetwork(), device_ids=args.use_gpus).to(device)
        print("Using multiple GPUs:", args.use_gpus)
    elif device == 'cuda' and len(args.use_gpus) == 1:
        infer_action = ClassificationNetwork().to(device)
        print("Using single GPU:", args.use_gpus[0])
    else:
        infer_action = ClassificationNetwork().to(device)
        print("Using CPU")
    
    # Define optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=True)


    # Constants
    nr_epochs = 500
    best_loss = 10e10
    batch_size = 512
    start_time = time.time()

    # Model Name
    model_dir = os.path.dirname(args.save_path)
    if args.save_path.endswith('.pt') or args.save_path.endswith('.pth'):
        model_name = os.path.basename(args.save_path).split(".")[0]
    else:
        model_name = "UnnamedModel.pt"
    
    # Create directory for saving models and tensorboard logs
    create_dir(model_dir)
    create_dir(args.tb_path)

    # Tensorboard, Early Stopping and Dataloader
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(args.tb_path, f'{model_name}_{timestamp}'))
    early_stopping = EarlyStopping(patience=10, mode='lowest') # Early Stopping if loss does not increase
    train_loader = DataLoader(
        CarlaDataset(args.image_dir, args.labels_path, image_size=args.image_size), 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers
    )
    
    # Load Weights if specified
    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        infer_action.load_state_dict(torch.load(args.load_model_path).state_dict())
        print(f"Loaded Weights From: {args.load_model_path}")

    
    # Training Loop
    for epoch in range(args.epochs):
        total_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), disable=args.disable_tqdm):
            batch_in, batch_gt = batch[0].to(device), batch[1].to(device)
            batch_out = infer_action(batch_in)
            loss = torch.nn.functional.mse_loss(batch_out, batch_gt)
            
            optimizer.zero_grad() 
            loss.backward()
            total_loss += loss

        scheduler.step(total_loss)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (args.epochs - 1 - epoch)
        
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA:" % (
            epoch + 1, total_loss), seconds_to_hms(time_left))

        # Write to Tensorboard
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler._last_lr[0], epoch)

        # Save Model and Early Stopping
        best_loss = save_model(model_name, model_dir, infer_action, total_loss, best_loss, mode="lowest") # Save Best and Last Model
        if early_stopping(total_loss, best_loss): # Early Stopping if loss does not increase
            print("Early Stopping")
            break

def create_dir(path):
    """
    Create a directory if it does not exist
    """
    # If the path is a file, get the directory
    if os.path.isfile(path):
        path = os.path.dirname(path)
    # Check if the directory exists, else create it
    if not os.path.exists(path):
        os.makedirs(path)

class EarlyStopping:
    def __init__(self, patience=10, mode='lowest'):
        self.patience = patience
        self.counter = 0
        self.mode = mode

    def __call__(self, val_metric, best_metric):
        self.counter += 1
        if (self.mode == 'lowest' and val_metric <= best_metric and self.counter > self.patience) or \
            (self.mode == 'highest' and val_metric >= best_metric and self.counter > self.patience):
            print("Early Stopping after %d epochs" % self.counter)
            return True

def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def save_model(name, model_dir, model, val_metric, best_metric, mode='lowest'):
    torch.save(model, os.path.join(model_dir, f'{name}_last.pt'))
    if mode == 'lowest':
        if val_metric <= best_metric:
            print(f"New {val_metric} <= Old {best_metric}. Saving: {f'{name}_best.pt'}")
            torch.save(model, os.path.join(model_dir, f'{name}_best.pt'))
            return val_metric
    elif mode == 'highest':
        if val_metric >= best_metric:
            print(f"New {val_metric} >= Old {best_metric}. Saving: {f'{name}_best.pt'}")
            torch.save(model, os.path.join(model_dir, f'{name}_best.pt'))
            return val_metric
    return best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-i', '--image_dir', default="../data/rgb", type=str, help='path to where you save the images you collect')
    parser.add_argument('-l', '--labels_path', default="../data/metrics/output.csv", type=str, help='path to where you save the labels you collect')
    parser.add_argument('-s', '--save_path', default="../models/", type=str, help='path where to save your model in .pt format')
    parser.add_argument('-t', '--tb_path', default="../tb_logs/", type=str, help='path where to save your Tensorboard Logs')
    parser.add_argument('-g', '--use_gpus', default=[0], type=int, nargs='+', help='List of GPU IDs to use')
    parser.add_argument('-is', '--image_size', default=[96, 96], type=int, nargs='+', help='Image Size (Width, Height)')
    parser.add_argument('-bs', '--batch_size', default=512, type=int, help='Batch Size')
    parser.add_argument('-nw', '--num_workers', default=16, type=int, help='Number of Workers for Dataloader')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning Rate')
    parser.add_argument('-wd', '--weight_decay', default=0.0, type=float, help='Weight Decay (Default, no weight decay)')
    parser.add_argument('-dt', '--disable_tqdm', action='store_true', help='Disable tqdm progress bar')
    parser.add_argument('-p', '--load_model_path', default=None, help='path to load your model in .pt format')
    args = parser.parse_args()
    
    train(args)
