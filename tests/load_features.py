import dill
import os
import sys

sys.path.insert(0, '..')

with open('data/features_pipeline.dill', 'rb') as f:
    pipeline = dill.load(f)

print(pipeline)
# Expected output: Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=128, ngram_range=(1, 2))),