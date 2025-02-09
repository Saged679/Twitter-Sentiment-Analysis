# Sentiment Analysis on Twitter Data
This project focuses on performing sentiment analysis on Twitter data to classify tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

# Table of Contents
Project Overview

Dataset

Project Steps

Installation

Usage

Results

Contributing

License

# Project Overview
The goal of this project is to analyze Twitter data and predict the sentiment of tweets using machine learning models. The project includes:

Loading and preprocessing the dataset.

Exploratory data analysis to understand the distribution of sentiments and text characteristics.

Feature engineering using TF-IDF vectorization.

Training and evaluating multiple machine learning models.

Hyperparameter tuning for the best-performing model.

Saving the trained model and vectorizer for future use.

# Dataset
The dataset consists of two CSV files:

twitter_training.csv: Training data containing tweets and their corresponding sentiments.

twitter_validation.csv: Validation data for testing the model.

Each dataset contains the following columns:

id: Unique identifier for each tweet.

entity: The entity or topic associated with the tweet.

sentiment: The sentiment label (Positive, Negative, Neutral, Irrelevant).

text: The content of the tweet.

# Project Steps
The project is divided into the following steps:

Data Loading and Exploration:

Load the datasets and inspect their structure.

Combine the training and validation datasets.

Handle missing values and duplicates.

Text Preprocessing:

Normalize text (lowercase, remove punctuation, stopwords, etc.).

Remove noise (URLs, HTML tags, special characters).

Exploratory Data Analysis (EDA):

Analyze the distribution of sentiments.

Visualize text length by sentiment.

Generate word clouds for each sentiment category.

Extract and visualize top n-grams.

Feature Engineering:

Convert text data into numerical features using TF-IDF vectorization.

Encode sentiment labels into numerical values.

Model Training and Evaluation:

Split the data into training and testing sets.

Train multiple models (Logistic Regression, Random Forest, etc.).

Evaluate models using accuracy, classification reports, and confusion matrices.

Hyperparameter Tuning:

Perform randomized search to find the best hyperparameters for the Random Forest model.

Save the Model and Vectorizer:

Save the trained model and TF-IDF vectorizer using joblib.

Visualization and Reporting:

Visualize model performance and key insights from EDA.

# Installation
To run this project, you need the following Python libraries:

numpy

pandas

matplotlib

seaborn

nltk

scikit-learn

wordcloud

joblib

You can install the required libraries using the following command:

pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud joblib
Additionally, download the necessary NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Usage
Clone the repository:

git clone https://github.com/Saged679/Twitter-Sentiment-Analysis.git
cd sentiment-analysis-twitter

Place the dataset files (twitter_training.csv and twitter_validation.csv) in the project directory.

Run the Jupyter Notebook or Python script:

jupyter notebook sentiment_analysis.ipynb
Follow the steps in the notebook to preprocess the data, train models, and evaluate performance.

# Results
Best Model: Random Forest with hyperparameter tuning achieved the highest accuracy.

Key Insights:

The dataset is imbalanced, with some sentiment categories having fewer samples.

Word clouds and n-grams provide insights into the most common words for each sentiment.

Text length varies significantly across sentiment categories.

# Contributing
Contributions to this project are welcome! If you have suggestions or improvements, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Commit your changes.

Submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, please contact:

Saged Ahmed

Email: saged5630@gmail.com

GitHub: Saged679

Thank you for using this project! 🚀
