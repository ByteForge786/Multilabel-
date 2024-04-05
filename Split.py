import csv

# Open the input CSV file and create output CSV files for test and train data
with open('your_csv_file.csv', 'r') as infile, \
        open('test.csv', 'w', newline='') as testfile, \
        open('train.csv', 'w', newline='') as trainfile:

    csvreader = csv.reader(infile)
    testwriter = csv.writer(testfile)
    trainwriter = csv.writer(trainfile)

    # Write headers to output files
    header = next(csvreader)
    testwriter.writerow(header)
    trainwriter.writerow(header)

    # Iterate through each row and split into test and train based on condition
    for row in csvreader:
        if all(int(value) == 0 for value in row[5:]):  # Check if values from column 6 onwards are all zero
            testwriter.writerow(row)
        else:
            trainwriter.writerow(row)
