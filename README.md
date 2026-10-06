# News Article Classifier

A text classification project that predicts the topic of a news article using TF-IDF and machine learning models.

## Dataset
20 Newsgroups (loaded automatically with scikit-learn), 6 categories:
comp.graphics, rec.autos, rec.sport.baseball, sci.med, sci.space, talk.politics.guns

- Train: 3508 articles
- Test: 2336 articles

## Method
1. Load data (headers, footers and quotes removed)
2. Convert text to numbers using TF-IDF (unigrams + bigrams, English stop words removed)
3. Train and compare 4 models: Logistic Regression, Linear SVM, Naive Bayes, KNN
4. Evaluate using accuracy, macro F1, classification report and confusion matrix

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.855 | 0.855 |
| Linear SVM | 0.847 | 0.848 |
| Logistic Regression | 0.845 | 0.846 |
| KNN | 0.196 | 0.144 |

Best model: Naive Bayes (85.5% accuracy).

KNN performs poorly because TF-IDF vectors are high-dimensional and sparse. Using cosine distance improves it.

## How to run
Open Google Colab, paste the code from news_classifier.py and run it.
Or locally:

    pip install -r requirements.txt
    python news_classifier.py

## Future improvements
- BERT / sentence-transformer embeddings
- More categories
- Streamlit web app
