 import csv

# Read rule IDs from the text file into a set
with open('rule_ids.txt', 'r') as txtfile:
    rule_ids = set(txtfile.read().splitlines())

# Open the input CSV file and create output CSV files for data with and without rule IDs
with open('input_data.csv', 'r') as infile, \
        open('with_rule_id.csv', 'w', newline='') as with_file, \
        open('without_rule_id.csv', 'w', newline='') as without_file:

    csvreader = csv.reader(infile)
    with_writer = csv.writer(with_file)
    without_writer = csv.writer(without_file)

    # Write headers to output files
    header = next(csvreader)
    with_writer.writerow(header)
    without_writer.writerow(header)

    # Iterate through each row and split into with and without rule ID
    for row in csvreader:
        if row[0] in rule_ids:  # Assuming rule_id is in the first column, adjust if it's in a different column
            with_writer.writerow(row)
        else:
            without_writer.writerow(row)
