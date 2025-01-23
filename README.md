# NLP-Natural-Language-Processing-

# Yelp Business Rating Prediction Project

Welcome to the **Yelp Business Rating Prediction Project**, where we leverage the power of Natural Language Processing (NLP) to classify Yelp reviews into 1-star or 5-star ratings based on the text content.

## Project Overview
This project demonstrates:
- Data exploration and visualization to uncover patterns in Yelp reviews.
- Text preprocessing and feature engineering using **CountVectorizer** and **TF-IDF**.
- Building classification models with **Multinomial Naive Bayes**.
- Evaluation of model performance using metrics such as accuracy, precision, recall, and F1-score.

The dataset used for this project is the **Yelp Review Data Set** from Kaggle.

## Dataset Details
Each observation in the dataset represents:
- A review of a business by a user.
- "Stars" column: The number of stars (1-5) assigned to the business by the reviewer.
- Additional columns: "cool", "useful", and "funny" votes for each review.

## Key Features of the Dataset
- **Stars:** Business rating (target variable).
- **Text:** Content of the review (input variable).
- **Cool, Useful, Funny Votes:** User feedback metrics.

## Workflow
1. **Exploratory Data Analysis (EDA):**
   - Explored text length vs. star ratings.
   - Visualized rating distribution using histograms, boxplots, and count plots.

2. **Text Processing and Feature Extraction:**
   - Preprocessed review text using:
     - **CountVectorizer:** Converts text into token counts.
     - **TF-IDF Transformer:** Converts token counts into weighted scores.

3. **Model Building and Training:**
   - Used **Multinomial Naive Bayes** for classification.
   - Created pipelines for seamless integration of preprocessing and modeling.

4. **Evaluation:**
   - Evaluated model performance on a test set using:
     - Confusion Matrix.
     - Classification Report.
   - Achieved an accuracy of **93%** with Naive Bayes.

## Code Example
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Creating pipeline
pipeline = Pipeline([
    ('cv', CountVectorizer()),
    ('tft', TfidfTransformer()),
    ('nb', MultinomialNB())
])

# Train-test split
X = data['text']
y = data['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Model training
pipeline.fit(X_train, y_train)

# Model evaluation
pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

## Results
- **Confusion Matrix:**
  ```
  [[159  69]
   [ 22 976]]
  ```
- **Classification Report:**
  - Precision: 93% (weighted average).
  - Recall: 93% (weighted average).
  - F1-score: 92% (weighted average).

## Visualizations
- Text length vs. star ratings:
  ```python
  sns.boxplot(data=data, y='text length', x='stars', palette='rainbow')
  ```

- Star rating distribution:
  ```python
  sns.countplot(data=data, x='stars', palette='rainbow')
  ```

## Tools and Libraries
- **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn.
- **Dataset Source:** [Yelp Dataset on Kaggle](https://www.kaggle.com).

## Learnings and Insights
- Reviews with higher ratings tend to have shorter text on average.
- Text preprocessing and feature engineering are critical for achieving high performance in NLP tasks.
- Naive Bayes performed well with text data and simple preprocessing.

## Next Steps
- Experiment with additional classifiers like Logistic Regression or Random Forest.
- Incorporate sentiment analysis for deeper insights into text reviews.
- Extend the classification to multi-class ratings (1-5 stars).

---
### Contact
For more details, feel free to reach out:
- **LinkedIn:** [Monirul Islam](https://www.linkedin.com/in/monirul-m08/)
- **GitHub:** [Monirul's GitHub](https://github.com/MonirulIslamm08)
- **Email:** md08monirul@gmail.com

---
### Acknowledgment
Thank you to the Kaggle community for providing the dataset and resources to learn and explore NLP techniques.
