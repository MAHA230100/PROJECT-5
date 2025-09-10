from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import save_joblib


def build_vectorizer(max_features: int = 50000, ngram_range=(1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=2
    )


def fit_transform_tfidf(df: pd.DataFrame, text_col: str = "text_blob", save_name: str = "tfidf") -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = build_vectorizer()
    X = vec.fit_transform(df[text_col].fillna(""))
    save_joblib(vec, save_name)
    return vec, X


def compute_similarity_matrix(X) -> np.ndarray:
    # Note: dense similarity can be large; we keep as float32
    sim = cosine_similarity(X, dense_output=False)
    return sim

