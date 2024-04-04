import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Step 1: Read the Excel file
data_excel = pd.read_excel('your_excel_file.xlsx')

# Step 2: Combine specified columns into 'information_context' column
data_excel['information_context'] = (
    "rule_name : " + data_excel['rule_name'].astype(str) + "\n"
    + "rule_sql_filter : " + data_excel['rule_sql_filter'].astype(str) + "\n"
    + data_excel['dataset_query'].astype(str)
)

# Step 3: Drop unnecessary columns
data_excel.drop(columns=['dataset_name', 'rule_name', 'rule_sql_filter', 'dataset_query'], inplace=True)

# Step 4: Rename columns
data_excel.columns = ['rule_id', 'information_context', 'owner_name', 'label']

# Step 5: One-hot encode the 'label' column
encoder = OneHotEncoder(sparse=False)
labels_one_hot = encoder.fit_transform(data_excel[['label']])
label_columns = [f'label_{i}' for i in range(labels_one_hot.shape[1])]
data_encoded = pd.concat([data_excel, pd.DataFrame(labels_one_hot, columns=label_columns)], axis=1)
data_encoded.drop(columns=['label'], inplace=True)

# Save the processed data to a new Excel file
data_encoded.to_excel('processed_data.xlsx', index=False)
