import os
import random
import functools
import csv
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


def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


# define custom batch preprocessor
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


# define which metrics to compute for evaluation
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


# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss


# set random seed
random.seed(0)

# load data
with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))
    header_row = data.pop(0)

# shuffle data
random.shuffle(data)

# reshape
idx, text, labels = list(zip(*[(int(row[0]), f'Title: {row[1].strip()}\n\nAbstract: {row[2].strip()}', row[3:]) for row in data]))
labels = np.array(labels, dtype=int)

# create label weights
label_weights = 1 - labels.sum(axis=0) / labels.sum()

# stratified train test split for multilabel ds
row_ids = np.arange(len(labels))
train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:, np.newaxis], labels, test_size=0.1)
x_train = [text[i] for i in train_idx.flatten()]
x_val = [text[i] for i in val_idx.flatten()]

# create hf dataset
ds = DatasetDict({
    'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
    'val': Dataset.from_dict({'text': x_val, 'labels': y_val})
})

# model name
model_name = 'mistralai/Mistral-7B-v0.1'

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds = tokenized_ds.with_format('torch')

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=labels.shape[1]
)
model.config.pad_token_id = tokenizer.pad_token_id

# define training args
training_args = TrainingArguments(
    output_dir='multilabel_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=8,  # tested with 16gb gpu ram
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# train
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

# save model
model_id = 'multilabel_mistral_no_quantization'
trainer.model.save_pretrained(model_id)
tokenizer.save_pretrained(model_id)

# load model
model = AutoModelForSequenceClassification.from_pretrained(model_id)
