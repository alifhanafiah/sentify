import nltk
import os
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# # Run once for nltk
# nltk.download("punkt")
# nltk.download("stopwords")

# get indonesian stopwords
list_stopwords = stopwords.words("indonesian")

# add additional stopwords
list_stopwords.extend(["haaaalaaaah", "emangtidak", "bagaiaman", "ajah"])

# add stopword from txt file
txt_stopword = pd.read_csv(
    os.path.join("database", "stopwords.txt"), names=["stopwords"], header=None
)
# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(" "))
# To make it faster
list_stopwords = set(list_stopwords)


# load stemmed term
with open(os.path.join("database", "term_dict.pkl"), "rb") as file:
    term_dict = pickle.load(file)


# case folding
def case_folding(text):
    return text.lower()


# cleaning
def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = (
        text.replace("\\t", " ")
        .replace("\\n", " ")
        .replace("\\u", " ")
        .replace("\\", "")
    )
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode("ascii", "replace").decode("ascii")
    # remove mention, link, hashtag
    text = " ".join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")


def remove_number(text):
    return re.sub(r"\d+", "", text)


def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)


# remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()


# remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub("\s+", " ", text)


# remove single char
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


# NLTK word rokenize
def word_tokenize_wrapper(text):
    return word_tokenize(text)


# remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


# apply stemming
def get_stemmed_term(document):
    # return [term_dict.get(term, term) for term in document]
    stemmed_terms = [term_dict.get(term, None) for term in document]
    return [term for term in stemmed_terms if term is not None]
