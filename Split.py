import pandas as pd

# Read the Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Check if all values from column 6 to the last column are zero for each row
zero_rows = df.iloc[:, 5:].eq(0).all(axis=1)

# Split the data based on the condition
test_data = df[zero_rows]
train_data = df[~zero_rows]

# Write the data to new Excel files
test_data.to_excel('test.xlsx', index=False)
train_data.to_excel('train.xlsx', index=False)
