## **Deployed App on Streamlit link :-** [click here](https://yashkumbalkar-quora-question-pair-similarity-project-app-00ufq0.streamlit.app/)

# Quora Question Pair Similarity Project

### Overview :-

The Quora Question Pair Similarity project is a machine learning-based model that predicts whether two questions on the Quora platform are similar or not. 
This model uses natural language processing (NLP) techniques to compare two questions and assess their semantic similarity, providing a binary classification 
result: similar or not similar.

### Data Source :-

The dataset used for this project is sourced from Kaggle:

- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)


### Project Description :-

This project leverages a machine learning approach to solve the problem of question pair similarity. The goal is to determine whether two given questions are 
asking about the same thing, even if they are phrased differently. This is useful in various applications, such as:

- Identifying duplicate questions on Q&A platforms like Quora.
- Building more intelligent search engines.
- Improving recommendation systems.

### Conclusions after EDA :-

- `63.08%` of question pairs is not duplicate and `36.91%` is duplicate in Dataset.
- There are 537933 unique questions and 111780 questions getting repeated.
- Frequency of Most Repeated questions are less than 60 and very few questions are Repeated more than 60 times. You can see one question which have `id-2559` repeated 157 times.
- Average Number of Characters in question 1 is 59 and in question 2 is 60.
- Average Number of Words in question 1 is 10 and in question 2 is 11.
- If Both questions have less than 4 common words then probabilty of getting not duplicate is high.
- Duplicate question pairs have more total words than non-duplicate question pairs.
- If word share percentage is more than `15%` than the probabilty of duplicate question is high.
- If the first two words and last two words of both questions are not same then probability of non-duplicate is high.

### Features :-

- `Question pair similarity prediction`: The core feature is the ability to predict if two questions are semantically similar.
- `Preprocessing`: Includes tokenization, vectorization, and stop-word removal.
- `Model training`: The project utilizes advanced machine learning algorithms for training the model.
- `Evaluation`: Performance metrics such as accuracy score, confusion matrix to evaluate the model's effectiveness.

### Technologies Used :-

- `Python` - Programming language used for building the project.
- `Scikit-learn` - For preprocessing and other machine learning model building and training tasks.
- `pandas` - For handling and processing data.
- `NumPy` - For numerical operations.
- `NLTK` - For natural language processing tasks like tokenization and stop word removal.
- `Matplotlib/Seaborn` - For data visualization and plotting results.

### Result :-

| Model                         | Vectorization Technique	 | Features      | Accuracy |
|-------------------------------|--------------------------|---------------|----------|
| Random Forest	                | Bag of Words             | Basic         | 76.85%   |
| XGBoost                       | Bag of Words             | Basic         | 76.48%   |
| Random Forest	                | Tfidf                    | Basic         | 76.71%   |
| XGBoost                       | Tfidf                    | Basic         | 76.23%   |
| Random Forest	                | Word2Vec	               | Basic         | 74.56%   |
| XGBoost                       | Word2Vec	               | Basic         | 75.7%    |
| Random Forest                 | Bag of Words             | Basic+Advance | 79.09%   |
| XGBoost                       | Bag of Words             | Basic+Advance | 79.52%   |
| Random Forest                 | Word2Vec                 | Basic+Advance | 78.54%   |
| XGBoost                       | Word2Vec	               | Basic+Advance | 78.54%   |

