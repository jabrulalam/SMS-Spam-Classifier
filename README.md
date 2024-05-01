# SMS-Spam-Mail-Classifier

This repository contains code for a spam mail classifier. The classifier processes text data, performs exploratory data analysis (EDA), preprocesses the data, builds classification models using various algorithms, evaluates model performance, and provides functionality for making predictions on new data.

The main components of the project include:

1.Data Cleaning:
The dataset is cleaned by handling missing values, removing duplicates, and transforming categorical labels into numerical format.

2.Exploratory Data Analysis (EDA):
EDA is performed to gain insights into the distribution and characteristics of spam and ham messages.

3.Data Preprocessing: 
Text data is preprocessed by converting to lowercase, tokenizing, removing special characters, punctuation, stop words, and applying stemming.
4.Text Vectorization: 
TF-IDF vectorization is used to convert text data into numerical features.

5.Model Building: 
Classification models are built using various algorithms such as Naive Bayes, Logistic Regression, SVM, etc. Model performance is evaluated using accuracy and precision scores.

6.Model Improvement: 
Strategies for improving model performance are explored, such as parameter tuning and feature scaling.

7.Saving Models: The TF-IDF vectorizer and the best performing model are saved using pickle for future use.

8.Prediction: Functionality is provided to predict whether a given message is spam or ham using the trained model.

#Dependencies
Python 3.x, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, Wordcloud

#Explanation of code

1.Importing Libraries:
import numpy as np: Imports NumPy library and provides an alias np for easy referencing.
import pandas as pd: Imports Pandas library and provides an alias pd for easy referencing.

2.Reading Data:
df = pd.read_csv("spam_mail_data.csv"): Reads the data from a CSV file named "spam_mail_data.csv" into a Pandas DataFrame called df.

3.Data Cleaning:
df.info(): Provides information about the DataFrame like the column names, data types, and non-null counts.
df.describe(): Generates descriptive statistics about the DataFrame like mean, median, min, max, etc.
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1}): Maps the 'Category' column values from string labels ('ham' and 'spam') to numerical values (0 and 1).
df.isnull().sum(): Checks for missing values in the DataFrame.
df.duplicated().sum(): Checks for duplicate rows in the DataFrame.
df = df.drop_duplicates(keep="first"): Drops duplicate rows while keeping the first occurrence.
df.shape: Returns the dimensions of the DataFrame (number of rows and columns).

4.Exploratory Data Analysis (EDA):
Visualizes the distribution of spam vs. ham messages and explores characteristics like message length, number of words, and number of sentences.

5.Data Preprocessing:
Lowercases text, tokenizes, removes special characters, punctuation, stop words, and applies stemming.
Creates new columns like 'num_characters', 'num_words', 'num_sentences' based on the processed text data.

6.Text Vectorization:
Uses TF-IDF vectorization for converting text data into numerical features.

7.Model Building:
Splits the data into training and testing sets.
Utilizes various classification algorithms like Naive Bayes, Logistic Regression, SVM, etc., for model building and evaluation.
Evaluates model performance using accuracy and precision scores.
Selects the best performing algorithm based on precision scores.

8.Model Improvement:
Explores strategies for improving model performance like changing parameters and feature scaling.

9.Saving Models:
Saves the TF-IDF vectorizer and the best performing model using pickle for future use.

10.Prediction:
Defines functions for predicting whether a given message is spam or ham.
Evaluates the models using cross-validation and generates classification reports.

11.Prediction Example:
Provides an example of how to use the trained model to predict whether a given message is spam or ham.




