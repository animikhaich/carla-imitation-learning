import csv

def merge_csv(file1, file2, file3, output_file):
    with open(file1, 'r') as file1, open(file2, 'r') as file2, open(file3, 'r') as file3, open(output_file, 'w') as output:
        file1_reader = csv.reader(file1)
        file2_reader = csv.reader(file2)
        file3_reader = csv.reader(file3)
        
        output_writer = csv.writer(output)
        
        # Write header from first file
        header = next(file1_reader)
        output_writer.writerow(header)
        
        # Discard header from second file
        next(file2_reader)
        
        # Discard header from third file
        next(file3_reader)
        
        # Write data from first file
        for row in file1_reader:
            output_writer.writerow(row)
        
        # Write data from second file
        for row in file2_reader:
            output_writer.writerow(row)
        
        # Write data from third file
        for row in file3_reader:
            output_writer.writerow(row)

# Example usage
merge_csv(
    "data/3090ti-v1/metrics/metrics.csv", 
    "data/2080ti-v1/metrics/metrics.csv",
    "data/scc-v1/metrics/metrics.csv", 
    "data/combined/metrics.csv"
    )
