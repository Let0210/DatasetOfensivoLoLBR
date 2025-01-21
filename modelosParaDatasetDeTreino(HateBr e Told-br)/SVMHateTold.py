import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time


df = pd.read_csv('juncao hatebr e ToLD-BR binarios minusculo_undersampling.csv')

X = df['message']
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_vectorization = time.time()
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
end_vectorization = time.time()

svm_model = SVC(kernel='linear', random_state=42)
start_training = time.time()
svm_model.fit(X_train_tfidf, y_train)
end_training = time.time()

start_testing = time.time()
y_pred = svm_model.predict(X_test_tfidf)
end_testing = time.time()

print("Métricas de Avaliação (Conjunto de Teste):")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Ofensivo', 'Ofensivo'], yticklabels=['Não Ofensivo', 'Ofensivo'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

print("\nTempos de Processamento:")
print(f"Tempo de Vetorização: {end_vectorization - start_vectorization:.4f} segundos")
print(f"Tempo de Treinamento: {end_training - start_training:.4f} segundos")
print(f"Tempo de Teste: {end_testing - start_testing:.4f} segundos")
