import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')


file_path = 'juncao comentarios classificadosBertimbauPreProcessamento2.csv'
data = pd.read_csv(file_path)

qtd_por_classif = data['classification'].value_counts()
print("Qtd de comentarios não ofensivos (0):", qtd_por_classif.get(0, 0))
print("Qtd de comentarios ofensivos (1):", qtd_por_classif.get(1, 0))

offensive_comments = data[data['classification'] == 1]['message']

stop_words = set(stopwords.words('portuguese'))
stop_words.add('user') 
stop_words.add('vai')
stop_words.add('vao')
stop_words.add('vão')
stop_words.add('nao')
all_offensive_text = " ".join(offensive_comments.dropna()).lower()
filtered_words = [word for word in all_offensive_text.split() if word not in stop_words]

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
