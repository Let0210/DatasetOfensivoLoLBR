import os
import time
import pickle
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

df = pd.read_csv('juncao hatebr e ToLD-BR binarios minusculo_undersampling.csv')

X = df['message'].tolist()
y = df['classification'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

tokenized_train_path = 'tokenized_trainHateToldDistilBERT.pkl'
tokenized_test_path = 'tokenized_testHateToldDistilBERT.pkl'

class CommentDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        data = self.tokenized_data[index]
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long).detach(),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long).detach(),
            'labels': torch.tensor(data['labels'], dtype=torch.long).detach()
        }

def tokenize_and_save(texts, labels, tokenizer, max_len, save_path):
    tokenized_data = []
    for text, label in zip(texts, labels):
        encoding = tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        tokenized_data.append({
            'input_ids': encoding['input_ids'].squeeze(0).tolist(),
            'attention_mask': encoding['attention_mask'].squeeze(0).tolist(),
            'labels': label
        })
    with open(save_path, 'wb') as f:
        pickle.dump(tokenized_data, f)
    return tokenized_data

if os.path.exists(tokenized_train_path):
    with open(tokenized_train_path, 'rb') as f:
        tokenized_train_data = pickle.load(f)
else:
    tokenized_train_data = tokenize_and_save(X_train, y_train, tokenizer, max_len=128, save_path=tokenized_train_path)


if os.path.exists(tokenized_test_path):
    with open(tokenized_test_path, 'rb') as f:
        tokenized_test_data = pickle.load(f)
else:
    tokenized_test_data = tokenize_and_save(X_test, y_test, tokenizer, max_len=128, save_path=tokenized_test_path)


train_dataset = CommentDataset(tokenized_train_data)
test_dataset = CommentDataset(tokenized_test_data)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,  # Mais épocas
    per_device_train_batch_size=16,  # Tamanho do batch reduzido
    per_device_eval_batch_size=16,
    learning_rate=3e-5,  # Taxa de aprendizado menor
    weight_decay=0.01,  # Regularização L2
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=500,  # Logging mais frequente
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    seed=42
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


start_train_time = time.time()
trainer.train()
end_train_time = time.time()
print(f"Tempo de treinamento: {end_train_time - start_train_time:.2f} segundos")


start_eval_time = time.time()
results = trainer.evaluate()
end_eval_time = time.time()
print(f"Tempo de avaliação no conjunto de teste: {end_eval_time - start_eval_time:.2f} segundos")

print("Resultados no conjunto de teste:", results)
