from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _get_index(df: pd.DataFrame, title: str, author: Optional[str] = None) -> Optional[int]:
    title = (title or "").strip().lower()
    if author:
        author = author.strip().lower()
    matches = df[(df.get("book_name", "").str.lower() == title)]
    if author and not matches.empty:
        matches = matches[matches.get("author", "").str.lower() == author]
    if matches.empty:
        # fallback partial match
        matches = df[df.get("book_name", "").str.lower().str.contains(title, na=False)]
    if matches.empty:
        return None
    return int(matches.index[0])


def recommend_content_based(df: pd.DataFrame, X, query_title: str, query_author: Optional[str] = None, top_k: int = 10) -> pd.DataFrame:
    idx = _get_index(df, query_title, query_author)
    if idx is None:
        return pd.DataFrame(columns=df.columns)
    sims = cosine_similarity(X[idx], X, dense_output=False).toarray().ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != idx][:top_k]
    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]
    return recs


def recommend_within_cluster(df: pd.DataFrame, X, labels: np.ndarray, query_title: str, query_author: Optional[str] = None, top_k: int = 10) -> pd.DataFrame:
    idx = _get_index(df, query_title, query_author)
    if idx is None:
        return pd.DataFrame(columns=df.columns)
    labels_arr = np.asarray(labels)
    cluster = labels_arr[idx]
    in_cluster = np.where(labels_arr == cluster)[0]
    sims = cosine_similarity(X[idx], X[in_cluster], dense_output=False).toarray().ravel()
    order_local = np.argsort(-sims)
    order = [in_cluster[i] for i in order_local if in_cluster[i] != idx][:top_k]
    recs = df.iloc[order].copy()
    recs["similarity"] = cosine_similarity(X[idx], X[order], dense_output=False).toarray().ravel()
    recs["cluster"] = cluster
    return recs


def recommend_hybrid(df: pd.DataFrame, X, labels: np.ndarray, query_title: str, query_author: Optional[str] = None, top_k: int = 10) -> pd.DataFrame:
    content = recommend_content_based(df, X, query_title, query_author, top_k=50)
    if content.empty:
        return content
    idx = _get_index(df, query_title, query_author)
    labels_arr = np.asarray(labels)
    cluster = labels_arr[idx]
    # Score 1: content similarity (robust min-max scaling)
    s = content["similarity"]
    s_min = float(np.min(s.values))
    s_max = float(np.max(s.values))
    s1 = (s - s_min) / (s_max - s_min + 1e-9)
    # Score 2: same cluster boost (align indices via take)
    same_cluster_bool = (labels_arr.take(content.index) == cluster)
    s2 = 0.2 + 0.8 * same_cluster_bool.astype(float)
    # Score 3: popularity (reviews) and rating
    pop = df.loc[content.index, "number_of_reviews"].fillna(0)
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    rating = df.loc[content.index, "rating"].fillna(df["rating"].median())
    rating = (rating - rating.min()) / (rating.max() - rating.min() + 1e-9)

    score = 0.55 * s1 + 0.25 * s2 + 0.20 * (0.5 * pop + 0.5 * rating)
    content = content.assign(hybrid_score=score)
    content = content.sort_values("hybrid_score", ascending=False).head(top_k)
    return content

