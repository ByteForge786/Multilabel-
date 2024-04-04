 import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the Excel file
data_excel = pd.read_excel('your_excel_file.xlsx')

# Split and encode attribute names
encoder = LabelEncoder()
data_excel['attribute_name'] = data_excel['attribute_name'].apply(lambda x: [encoder.fit_transform([item.strip()])[0] for item in x.split(',')])

# Create binary labels based on the 'label' column
for index, row in data_excel.iterrows():
    labels = row['label'].split(', ')
    for label in labels:
        data_excel.at[index, label.strip()] = 1

# Drop unnecessary columns
data_encoded = data_excel.drop(columns=['attribute_name', 'label'])

# Save the processed data to a new Excel file
data_encoded.to_excel('processed_data.xlsx', index=False)
