import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Read the Excel file
data_excel = pd.read_excel('your_excel_file.xlsx')

# Define a lambda function to create a unique information_context for each row
create_information_context = lambda row: f"rule_name : {row['rule_name']}\nrule_sql_filter : {row['rule_sql_filter']}\n{row['dataset_query']}"
data_excel['information_context'] = data_excel.apply(create_information_context, axis=1)

# Drop unnecessary columns
data_excel.drop(columns=['dataset_name', 'rule_name', 'rule_sql_filter', 'dataset_query'], inplace=True)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
labels_one_hot = encoder.fit_transform(data_excel[['label']])
label_columns = [f'label_{i}' for i in range(labels_one_hot.shape[1])]
data_encoded = pd.concat([data_excel, pd.DataFrame(labels_one_hot, columns=label_columns)], axis=1)
data_encoded.drop(columns=['label'], inplace=True)

# Save the processed data to a new Excel file
data_encoded.to_excel('processed_data.xlsx', index=False)
