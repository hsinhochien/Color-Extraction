# pip install datasets
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline

FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open('./data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 分割数据集
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42) 
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 创建 Huggingface 数据集
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, is_split_into_words=False)
    
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their corresponding word.
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

label_all_tokens = False
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "Babelscape/wikineural-multilingual-ner",
    num_labels=2, 
    ignore_mismatched_sizes=True
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_true_labels = [item for sublist in true_labels for item in sublist]
    all_true_predictions = [item for sublist in true_predictions for item in sublist]
    
    precision = precision_score(all_true_labels, all_true_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_true_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_true_predictions, average='weighted')
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)
logging.info("Start training.")
trainer.train()

model.save_pretrained("./temp_model")
tokenizer.save_pretrained("./temp_model")
logging.info("Finish training.")

# 評估現有的最佳模型
existing_model = AutoModelForTokenClassification.from_pretrained("./best_model")
existing_tokenizer = AutoTokenizer.from_pretrained("./best_model")

test_dataset = test_dataset.map(lambda examples: existing_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, is_split_into_words=False), batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

trainer_existing = Trainer(
    model=existing_model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

logging.info("Evaluating existing best model on test dataset.")
existing_results = trainer_existing.evaluate()

# 測試新訓練的模型
best_model = AutoModelForTokenClassification.from_pretrained("./temp_model")
best_tokenizer = AutoTokenizer.from_pretrained("./temp_model")

test_dataset = test_dataset.map(lambda examples: best_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, is_split_into_words=False), batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

trainer_new = Trainer(
    model=best_model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

logging.info("Evaluating new model on test dataset.")
new_results = trainer_new.evaluate()

# 比较 F1 分数
if new_results['eval_f1'] > existing_results['eval_f1']:
    best_model.save_pretrained("./best_model")
    best_tokenizer.save_pretrained("./best_model")
    logging.info("New best model saved.")
else:
    logging.info("The existing best model is better or equal to the new model.")