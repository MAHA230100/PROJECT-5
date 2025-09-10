from typing import Tuple, List
import os
import re
import pandas as pd
import numpy as np
from .utils import normalize_text

DATASET1 = "Audible_Catlog.csv"
DATASET2 = "Audible_Catlog_Advanced_Features.csv"


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	renamed = {c: normalize_text(c).replace(" ", "_") for c in df.columns}
	df = df.rename(columns=renamed)
	return df


def _parse_numeric(s):
	if pd.isna(s):
		return np.nan
	if isinstance(s, (int, float)):
		return float(s)
	s = str(s)
	s = re.sub(r"[^0-9.]+", "", s)
	try:
		return float(s)
	except Exception:
		return np.nan


def _infer_genre_from_text(row: pd.Series) -> str:
	# Simple keyword-based genre inference from description/title if genre missing
	genre_text = str(row.get("genre", "")).strip().lower()
	desc = " ".join([
		str(row.get("book_name", "")),
		str(row.get("description", "")),
		str(row.get("ranks", "")),
	]).lower()
	if genre_text:
		return genre_text
	keywords = {
		"science fiction": ["science fiction", "sci-fi", "sci fi", "sf", "space opera", "dystopian"],
		"fantasy": ["fantasy", "sword", "dragon", "magic"],
		"thriller": ["thriller", "suspense", "psychological thriller"],
		"mystery": ["mystery", "detective", "whodunit", "crime"],
		"romance": ["romance", "love story", "romantic"],
		"non-fiction": ["nonfiction", "non-fiction", "biography", "memoir", "self-help", "history"],
		"horror": ["horror", "ghost", "haunted", "supernatural"],
		"young adult": ["young adult", "ya"],
		"history": ["history", "historical"],
	}
	found = []
	for g, toks in keywords.items():
		for t in toks:
			if t in desc:
				found.append(g)
				break
	return ", ".join(sorted(set(found)))


def _infer_year_from_text(row: pd.Series) -> float:
	if not pd.isna(row.get("publication_year", np.nan)):
		return row.get("publication_year")
	text = " ".join([
		str(row.get("ranks", "")),
		str(row.get("description", "")),
		str(row.get("book_name", "")),
	]).lower()
	m = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
	for cand in m:
		year = pd.to_numeric(cand, errors="coerce")
		if pd.notna(year) and 1900 <= int(year) <= 2025:
			return float(year)
	return np.nan


def load_raw(root: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
	p1 = os.path.join(root, DATASET1)
	p2 = os.path.join(root, DATASET2)
	df1 = pd.read_csv(p1, encoding="utf-8", low_memory=False)
	df2 = pd.read_csv(p2, encoding="utf-8", low_memory=False)
	return _standardize_columns(df1), _standardize_columns(df2)


def clean_and_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
	for df in (df1, df2):
		if "book_name" in df:
			df["book_name"] = df["book_name"].map(normalize_text)
		if "author" in df:
			df["author"] = df["author"].map(normalize_text)
		if "genre" in df:
			df["genre"] = df["genre"].fillna("").astype(str).map(lambda x: ", ".join(sorted(set([g.strip().lower() for g in re.split(r"[,/|]", x) if g.strip()]))))
		if "rating" in df:
			df["rating"] = df["rating"].map(_parse_numeric)
		if "number_of_reviews" in df:
			df["number_of_reviews"] = df["number_of_reviews"].map(_parse_numeric)
		if "price" in df:
			df["price"] = df["price"].map(_parse_numeric)
		if "publication_year" in df:
			df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")

	keys = [k for k in ["book_name", "author"] if k in df1.columns and k in df2.columns]
	if not keys:
		keys = ["book_name"] if "book_name" in df1.columns and "book_name" in df2.columns else []

	merged = None
	if keys:
		merged = pd.merge(df1, df2, on=keys, how="outer", suffixes=("_1", "_2"))
	else:
		merged = pd.concat([df1, df2], axis=0, ignore_index=True)

	# Coalesce common fields
	def coalesce(cols: List[str]):
		vals = [merged[c] for c in cols if c in merged]
		if not vals:
			return None
		out = vals[0].copy()
		for v in vals[1:]:
			out = out.combine_first(v)
		return out

	for field in ["rating", "number_of_reviews", "price", "genre", "description", "listening_time", "ranks", "publication_year"]:
		cols = [c for c in merged.columns if re.fullmatch(fr"{field}(?:_[12])?", c)]
		series = coalesce(cols)
		if series is not None:
			merged[field] = series

	# Drop helper dup columns after coalesce
	drop_cols = [c for c in merged.columns if re.search(r"_(1|2)$", c)]
	merged = merged.drop(columns=drop_cols, errors="ignore")

	merged = merged.drop_duplicates(subset=[c for c in ["book_name", "author"] if c in merged], keep="first")

	# Impute missing
	if "rating" in merged:
		merged["rating"] = merged["rating"].fillna(merged["rating"].median())
	if "number_of_reviews" in merged:
		merged["number_of_reviews"] = merged["number_of_reviews"].fillna(0)
	if "price" in merged:
		merged["price"] = merged["price"].fillna(merged["price"].median())

	# Feature helpers
	if "description" not in merged:
		merged["description"] = ""
	if "genre" not in merged:
		merged["genre"] = ""

	# Infer genre if empty
	merged["genre"] = merged.apply(_infer_genre_from_text, axis=1)
	# Normalize genre list formatting again (unique, sorted)
	merged["genre"] = merged["genre"].fillna("").astype(str).map(lambda x: ", ".join(sorted(set([g.strip().lower() for g in re.split(r"[,/|]", x) if g.strip()]))))

	# Infer publication year if missing
	merged["publication_year"] = merged.apply(_infer_year_from_text, axis=1)

	merged["text_blob"] = (
		merged.get("book_name", pd.Series([""] * len(merged))).astype(str)
		+ " "
		+ merged.get("author", pd.Series([""] * len(merged))).astype(str)
		+ " "
		+ merged.get("genre", pd.Series([""] * len(merged))).astype(str)
		+ " "
		+ merged.get("description", pd.Series([""] * len(merged))).astype(str)
	).map(lambda s: s.lower())

	return merged


def basic_eda(df: pd.DataFrame) -> dict:
	out = {
		"n_rows": int(df.shape[0]),
		"n_cols": int(df.shape[1]),
		"top_genres": df["genre"].fillna("").str.get_dummies(sep=",").sum().sort_values(ascending=False).head(20).to_dict() if "genre" in df else {},
		"top_authors": df.get("author", pd.Series(dtype=str)).value_counts().head(20).to_dict(),
		"rating_stats": df.get("rating", pd.Series(dtype=float)).describe().to_dict(),
		"reviews_correlation": float(df[["rating", "number_of_reviews"]].corr().iloc[0,1]) if set(["rating","number_of_reviews"]).issubset(df.columns) else np.nan,
	}
	if "publication_year" in df:
		out["year_counts"] = df["publication_year"].dropna().astype(int).value_counts().sort_index().to_dict()
	return out

