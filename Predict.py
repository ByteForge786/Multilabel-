import csv
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the test dataset
with open('test.csv', newline='') as csvfile:
    test_data = list(csv.reader(csvfile, delimiter=','))

# Extract text from test data
test_text = [f'Title: {row[1].strip()}\n\nAbstract: {row[2].strip()}' for row in test_data]

# Tokenize the test data
tokenizer = AutoTokenizer.from_pretrained('multilabel_mistral_no_quantization')
tokenizer.pad_token = tokenizer.eos_token
tokenized_test = tokenizer(test_text, padding=True, truncation=True, return_tensors='pt')

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained('multilabel_mistral_no_quantization')

# Batch prediction
batch_size = 8
num_batches = (len(test_text) + batch_size - 1) // batch_size

# Perform inference batch by batch
with torch.no_grad(), open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id'] + ['Label_' + str(i) for i in range(model.config.num_labels)])

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_text))

        batch_inputs = {key: val[start_idx:end_idx] for key, val in tokenized_test.items()}
        batch_outputs = model(**batch_inputs)

        # Convert logits to predictions
        predictions = torch.sigmoid(batch_outputs.logits)

        # Write predictions to CSV
        for i, pred in enumerate(predictions):
            writer.writerow([start_idx + i] + pred.tolist())
