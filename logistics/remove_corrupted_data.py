import os
import csv
import cv2
from tqdm import tqdm

# Folder containing the images
folder = '/projectnb/rlvn/animikh/carla-imitation-learning/data/small-images/rgb'

# Path to the CSV file
csv_file = '/projectnb/rlvn/animikh/carla-imitation-learning/data/small-images/metrics/metrics.csv'

# Open the CSV file for reading and writing
with open(csv_file, 'r') as input_file, open(csv_file + '.tmp', 'w', newline='') as output_file:
    reader = csv.DictReader(input_file)
    writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
    writer.writeheader()

    # Loop over each row in the CSV file
    for row in tqdm(reader, total=len(os.listdir(folder))):
        filename = row['filename']

        # Check if the file is a JPG image
        if filename.lower().endswith('.jpg'):
            filepath = os.path.join(folder, filename)

            # Try to open the image file
            try:
                img = cv2.imread(filepath)
                cv2.imwrite(filepath, img)
                # The image is not corrupt, so write the row to the output file
                writer.writerow(row)
            except:
                # The image is corrupt, so delete the file and the row
                os.remove(filepath)
                print(f"Deleted {filepath} because it is corrupt")
        else:
            # The file is not a JPG image, so write the row to the output file
            writer.writerow(row)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Replace the original CSV file with the temporary file
os.replace(csv_file + '.tmp', csv_file)
