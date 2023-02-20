import os
import shutil
import argparse
from tqdm import tqdm

def copy_files(src_folder, dst_folder):
    """
    Copies all files from the source folder to the destination folder.
    Displays a progress bar using tqdm.
    """
    # Create the destination folder if it doesn't already exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Get a list of all files in the source folder
    file_list = os.listdir(src_folder)

    # Copy each file from the source folder to the destination folder
    for file_name in tqdm(file_list, desc='Copying files', ascii=True):
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Copy files from one folder to another')

    # Add the arguments for source and destination folders
    parser.add_argument('--src', help='path to the source folder')
    parser.add_argument('--dst', help='path to the destination folder')

    # Parse the arguments
    args = parser.parse_args()

    # Call the copy_files function with the source and destination folders
    copy_files(args.src, args.dst)
