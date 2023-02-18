import csv
import argparse

def merge_csv(files, output_file):
    with open(output_file, 'w') as output:
        output_writer = csv.writer(output)

        # Merge headers from all input files
        headers = set()
        for file in files:
            with open(file, 'r') as input_file:
                file_reader = csv.reader(input_file)
                headers.add(tuple(next(file_reader)))
        if len(headers) != 1:
            raise ValueError('All input files should have the same headers')
        output_writer.writerow(headers.pop())

        # Merge data from all input files
        for file in files:
            with open(file, 'r') as input_file:
                file_reader = csv.reader(input_file)
                # Discard header
                next(file_reader)
                for row in file_reader:
                    output_writer.writerow(row)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Merge CSV files into one')
    
    # Add the arguments for input files and output file
    parser.add_argument('--files', nargs='+', help='paths to input files')
    parser.add_argument('--output_file', help='path to the output file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the merge_csv function with the input file paths and output file path
    merge_csv(args.files, args.output_file)
