import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def todense(X):
    return np.asarray(X.todense())

def reason_pipeline_steps():
    steps = [
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            max_features=128,
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )),
        # ('tfidf', TfidfVectorizer()),
        ('todense', FunctionTransformer(todense, validate=False))
    ]
    return steps

def numeric_pipeline_steps():
    steps = [
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ]
    return steps