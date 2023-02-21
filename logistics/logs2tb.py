import re
import torch
from torch.utils.tensorboard import SummaryWriter

# Create a summary writer to write to TensorBoard
writer = SummaryWriter("/projectnb/rlvn/animikh/carla-imitation-learning/tb_logs/classifier_small_v1/classifier_small_v1_20-02-2023_08-36-08")

# Open the input file
with open('/projectnb/rlvn/animikh/carla-imitation-learning/classifier_small/clf_small.o425661', 'r') as f:

    # Loop through each line
    for line in f:

        # Use a regular expression to extract the epoch and loss value from the line
        match = re.search(r'Epoch\s+(\d+)\s+\[Train\]\s+loss:\s+(\d+\.\d+)', line)

        # If there is a match, write the data to TensorBoard
        if match:

            epoch = int(match.group(1))
            loss = float(match.group(2))

            writer.add_scalar('Loss/train', loss, epoch)

# Close the summary writer
writer.close()