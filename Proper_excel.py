 import pandas as pd

# Read the Excel file into a pandas DataFrame
df = pd.read_excel("your_excel_file.xlsx")

# Pivot the DataFrame to get unique attribute_name as columns and is_cde as values
pivot_df = df.pivot_table(index='rule_id', columns='attribute_name', values='is_cde', aggfunc='first')

# Reset index to make 'rule_id' a column again
pivot_df.reset_index(inplace=True)

# Merge the pivoted DataFrame with the original DataFrame to include other columns
merged_df = pd.merge(df.drop(columns='attribute_name'), pivot_df, on='rule_id')

# Fill NaN values with 0
merged_df.fillna(0, inplace=True)

# Save the merged DataFrame as an Excel file
merged_df.to_excel("output_file.xlsx", index=False)
