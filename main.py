
# //import streamlit as st: Imports the Streamlit library and provides an alias st for easy referencing.
# //import pickle: Imports the pickle module for loading the pre-trained models.
# //import string: Imports the string module for handling string operations.
# //import nltk: Imports the Natural Language Toolkit (NLTK) library for text processing.
# //from nltk.corpus import stopwords: Imports the stopwords corpus from NLTK, which contains common words that are often removed during text processing.
# //from nltk.stem.porter import PorterStemmer: Imports the PorterStemmer class from NLTK, which is used for stemming words.

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Text Preprocessing Function (transform_text):
# Defines a function transform_text that takes a text input, preprocesses it, and returns the preprocessed text.
# Converts the text to lowercase.
# Tokenizes the text into individual words.
# Removes non-alphanumeric characters from each word.
# Removes stopwords (commonly occurring words like 'the', 'is', 'and') and punctuation marks.
# Stems each word using the Porter stemming algorithm to reduce words to their root form.
 
ps = PorterStemmer()
def transform_text(text):
 text = text.lower()
 text = nltk.word_tokenize(text)

 y = []
 for i in text:
  if i.isalnum():
   y.append(i)

 text = y[:]
 y.clear()

 for i in text:
  if i not in stopwords.words('english') and i not in string.punctuation:
   y.append(i)

 text = y[:]
 y.clear()

 for i in text:
  y.append(ps.stem(i))

 return " ".join(y)


# Loading Pre-trained Models:
# tfidf = pickle.load(open('vectorizer.pkl', 'rb')): Loads the TF-IDF vectorizer from the saved pickle file 'vectorizer.pkl'.
# model = pickle.load(open('model.pkl', 'rb')): Loads the pre-trained classification model from the saved pickle file 'model.pkl'.

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App:
# st.title("Email/SMS Spam Classifier"): Sets the title of the Streamlit app.
# input_sms = st.text_input("Enter the message"): Creates a text input box where the user can enter the email or SMS message.
# if st.button('Predict'):: Checks if the user has clicked the 'Predict' button.
# transformed_sms = transform_text(input_sms): Preprocesses the input message using the transform_text function defined earlier.
# vector_input = tfidf.transform([transformed_sms]): Vectorizes the preprocessed message using the loaded TF-IDF vectorizer.
# result = model.predict(vector_input)[0]: Uses the pre-trained model to predict whether the message is spam or not.
# Depending on the prediction result, it displays either "Spam" or "Not Spam" using st.header.

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

  #1.preprocess
  transformed_sms = transform_text(input_sms)
  #2.vectorize
  vector_input = tfidf.transform([transformed_sms])
  #3.predict
  result = model.predict(vector_input)[0]
  #4.display
  if result == 1:
   st.header("Spam")
  else:
   st.header("Not Spam")

