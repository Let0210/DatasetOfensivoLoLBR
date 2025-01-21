import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time


df = pd.read_csv('juncao hatebr e ToLD-BR binarios minusculo_undersampling.csv')

X = df['message']
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

start_train_time = time.time()
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
end_train_time = time.time()
train_time = end_train_time - start_train_time

start_test_time = time.time()
y_pred = rf_model.predict(X_test_tfidf)
end_test_time = time.time()
test_time = end_test_time - start_test_time

print("Métricas de Avaliação:")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nTempo de Treinamento do Modelo: {:.2f} segundos".format(train_time))
print("Tempo de Teste do Modelo: {:.2f} segundos".format(test_time))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Ofensivo', 'Ofensivo'], yticklabels=['Não Ofensivo', 'Ofensivo'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
