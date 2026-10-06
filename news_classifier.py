# News Article Classifier
# NLP project: TF-IDF + Logistic Regression, Linear SVM, Naive Bayes, KNN
# Dataset: 20 Newsgroups (6 categories), auto-downloaded by scikit-learn

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 1. Load data (auto download, no login needed)
cats = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns', 'rec.autos']
rm = ('headers', 'footers', 'quotes')
tr = fetch_20newsgroups(subset='train', categories=cats, remove=rm)
te = fetch_20newsgroups(subset='test', categories=cats, remove=rm)
names = tr.target_names
print("Train:", len(tr.data), "Test:", len(te.data), "Classes:", names)
pd.Series([names[i] for i in tr.target]).value_counts().plot(kind='barh', figsize=(8,5), title='Documents per category')
plt.show()

# 2. Train and compare models
models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Linear SVM': LinearSVC(), 'Naive Bayes': MultinomialNB(), 'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine')}
fitted = {n: Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000, sublinear_tf=True)), ('clf', c)]).fit(tr.data, tr.target) for n, c in models.items()}
preds_all = {n: p.predict(te.data) for n, p in fitted.items()}
res = pd.DataFrame([{'Model': n, 'Accuracy': accuracy_score(te.target, p), 'Macro F1': f1_score(te.target, p, average='macro')} for n, p in preds_all.items()]).sort_values('Macro F1', ascending=False)
print("\nModel comparison:\n", res.to_string(index=False))

# 3. Evaluate best model
best = res.iloc[0]['Model']
print("\nBest model:", best)
print(classification_report(te.target, preds_all[best], target_names=names))
plt.figure(figsize=(9,7))
sns.heatmap(confusion_matrix(te.target, preds_all[best]), annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
plt.title('Confusion Matrix - ' + best); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.show()

# 4. Test with your own text
sample = "NASA launched a new rocket to study the moon and planets"
print("Prediction:", names[fitted[best].predict([sample])[0]])
