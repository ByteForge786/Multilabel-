import os
import random
import functools
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import pandas as pd

# Define functions and classes

def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['features'], truncation=True, padding=True)
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average='micro')
    f1_macro = f1_score(labels, predictions > 0, average='macro')
    f1_weighted = f1_score(labels, predictions > 0, average='weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss

# Set random seed
random.seed(0)

# Load data from Excel
data_excel = pd.read_excel('your_excel_file.xlsx')

# Shuffle the data
data_excel = data_excel.sample(frac=1, random_state=0).reset_index(drop=True)

# Reshape the data
idx, features, labels = list(zip(*[(row['id'], ' '.join(row[['feature1', 'feature2', 'feature3']]), row['labels']) for _, row in data_excel.iterrows()]))
labels = np.array(labels, dtype=int)

# Split data into train and validation
row_ids = np.arange(len(labels))
train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:, np.newaxis], labels, test_size=0.1)
x_train = [features[i] for i in train_idx.flatten()]
x_val = [features[i] for i in val_idx.flatten()]

# Create dataset
ds = DatasetDict({
    'train': Dataset.from_dict({'features': x_train, 'labels': y_train}),
    'val': Dataset.from_dict({'features': x_val, 'labels': y_val})
})

# Define tokenizer and tokenize dataset
model_name = 'your_model_name'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds = tokenized_ds.with_format('torch')

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=labels.shape[1]
)
model.config.pad_token_id = tokenizer.pad_token_id

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=500,
    save_total_limit=2,
)

# Train model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['val'],
    tokenizer=tokenizer,
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    label_weights=torch.tensor(label_weights, device=model.device)
)

trainer.train()
