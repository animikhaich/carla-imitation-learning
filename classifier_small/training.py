import time
import random
import argparse

import torch

from network import ClassificationNetwork
from dataset import get_dataloader

import time
from tqdm import tqdm


def train(data_folder, labels_path, save_path, resume=False):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    device_id = [0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if device == 'cuda':
    #     infer_action = torch.nn.DataParallel(ClassificationNetwork(), device_ids=device_id).to(device)
    # else:
    infer_action = ClassificationNetwork().to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    
    model_name = "all_250k_small_v3"
    nr_epochs = 300
    best_loss = 10e10
    batch_size = 512
    nr_of_classes = 0  # needs to be changed
    start_time = time.time()

    if resume:
        infer_action.load_state_dict(torch.load(f"{model_name}_best.pt").state_dict())

    train_loader = get_dataloader(data_folder, labels_path, batch_size, num_workers=16)

    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(device), batch[1].to(device)
            
            batch_out = infer_action(batch_in)

            loss = cross_entropy_loss(batch_out, batch_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

        best_loss = save_model(model_name, infer_action, total_loss, best_loss, mode="lowest")

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

def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    loss = torch.nn.CrossEntropyLoss()
    return loss(batch_out, batch_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./data/rgb", type=str, help='path to where you save the images you collect')
    parser.add_argument('-l', '--labels_path', default="./data/metrics/output.csv", type=str, help='path to where you save the labels you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    parser.add_argument('-r', '--resume', default=False, help='Resume Training', action='store_true')
    args = parser.parse_args()
    
    train(args.data_folder, args.labels_path, args.save_path, resume=args.resume)
