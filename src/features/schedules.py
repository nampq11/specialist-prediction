import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   StandardScaler)


def todense(X):
    return np.asarray(X.todense())

def reason_pipeline_steps():
    steps = [
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.5,
            max_features=5000,
        )),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        # ('tfidf', TfidfVectorizer(max_df=5, max_features=128, ngram_range=(1, 2)))
        # ('todense', FunctionTransformer(todense, validate=False))
    ]
    return steps

def numeric_pipeline_steps():
    steps = [
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ]
    return steps