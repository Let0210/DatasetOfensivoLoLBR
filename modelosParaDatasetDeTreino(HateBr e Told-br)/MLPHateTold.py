import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time


df = pd.read_csv('classificando/juncao hatebr e ToLD-BR binarios minusculo_undersampling.csv')

X = df['message']
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


start_training_time = time.time()  
mlp_model = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train_tfidf, y_train)
end_training_time = time.time()  

training_time = end_training_time - start_training_time
print(f"Tempo de treinamento do modelo: {training_time:.2f} segundos")

start_testing_time = time.time() 
y_pred = mlp_model.predict(X_test_tfidf)
end_testing_time = time.time()  

testing_time = end_testing_time - start_testing_time
print(f"Tempo de teste do modelo: {testing_time:.2f} segundos")

print("\nMétricas de Avaliação:")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Ofensivo', 'Ofensivo'], yticklabels=['Não Ofensivo', 'Ofensivo'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
