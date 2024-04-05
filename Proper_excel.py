import pandas as pd

# Define the chunk size
chunksize = 10000  # Adjust this value based on your available memory

# Initialize an empty list to store the processed chunks
chunks = []

# Read the Excel file in chunks
for chunk in pd.read_excel("your_excel_file.xlsx", chunksize=chunksize):
    # Pivot the chunk to get unique attribute_name as columns and is_cde as values
    pivot_df = chunk.pivot_table(index='rule_id', columns='attribute_name', values='is_cde', aggfunc='first')
    
    # Reset index to make 'rule_id' a column again
    pivot_df.reset_index(inplace=True)
    
    # Merge the pivoted chunk with the original chunk to include other columns
    merged_df = pd.merge(chunk.drop(columns='attribute_name'), pivot_df, on='rule_id')
    
    # Fill NaN values with 0
    merged_df.fillna(0, inplace=True)
    
    # Append the processed chunk to the list
    chunks.append(merged_df)

# Concatenate the processed chunks into a single DataFrame
final_df = pd.concat(chunks)

# Save the final DataFrame as an Excel file
final_df.to_excel("output_file.xlsx", index=False)
