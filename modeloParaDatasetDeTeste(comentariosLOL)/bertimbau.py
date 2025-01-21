import os
import time
import pickle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import pandas as pd

# Funções e Classes
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PreTokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        return self.tokenized_data[index]

def tokenize_and_save(texts, labels, tokenizer, max_len, save_path):
    dataset = CommentDataset(texts, labels, tokenizer, max_len)
    tokenized_data = [{'input_ids': data['input_ids'], 'attention_mask': data['attention_mask'], 'labels': data['labels']} for data in dataset]
    with open(save_path, 'wb') as f:
        pickle.dump(tokenized_data, f)
    return tokenized_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1).numpy()
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
    }


df_train = pd.read_csv('juncao hatebr e ToLD-BR binarios minusculo_undersampling.csv')
tokenized_train_path = 'tokenized_train.pkl'
tokenized_test_path = 'tokenized_test_juncao_comentarios_tratados.pkl'

X_train = df_train['message'].tolist()
y_train = df_train['classification'].tolist()

df_test = pd.read_csv('comentariosAClassificar.csv')

if 'classification' not in df_test.columns:
    df_test['classification'] = None

X_test = df_test['message'].tolist()
y_test = [0] * len(X_test)  

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

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

train_dataset = PreTokenizedDataset(tokenized_train_data)
test_dataset = PreTokenizedDataset(tokenized_test_data)

model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=2,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
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

predictions = trainer.predict(test_dataset).predictions
predicted_labels = torch.argmax(torch.tensor(predictions), axis=1).numpy()

df_test['classification'] = predicted_labels

df_test.to_csv('juncao comentarios tratados_atualizado.csv', index=False)
print("Classificações salvas no arquivo 'juncao comentarios tratados_atualizado.csv'.")
