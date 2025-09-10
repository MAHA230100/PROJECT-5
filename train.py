import os
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from src.utils import load_joblib, load_json
from src.recommenders import recommend_content_based, recommend_within_cluster, recommend_hybrid
from src.data import load_raw, clean_and_merge
from src.features import fit_transform_tfidf
from src.clustering import fit_kmeans

# Page setup
st.set_page_config(page_title="Book Recommender", layout="wide")
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .main {
    background: #f7fafc;
    font-family: 'Inter', sans-serif;
    color: #1f2937;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
    padding: 2rem 1rem;
    width: 260px;
}
.sidebar-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1f2937;
    text-align: center;
    margin-bottom: 2rem;
}
.nav-wrap > div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    border-radius: 8px;
    color: #374151;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    cursor: pointer;
    transition: all 0.2s ease;
}
.nav-wrap > div[role="radiogroup"] > label:hover {
    background: #e5e7eb;
    color: #111827;
}
.nav-wrap > div[role="radiogroup"] > label[data-checked="true"] {
    background: #3b82f6;
    color: #ffffff;
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}
.nav-wrap > div[role="radiogroup"] > label[data-checked="true"]::before {
    content: "";
    width: 5px;
    height: 60%;
    border-radius: 4px;
    background: #ffffff;
    opacity: 0.9;
}
.nav-wrap > div[role="radiogroup"] input[type="radio"],
.nav-wrap > div[role="radiogroup"] label svg,
.nav-wrap > div[role="radiogroup"] label div[aria-hidden="true"] {
    display: none !important;
}
input, textarea {
    background: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    padding: 0.75rem !important;
    font-size: 0.95rem !important;
    color: #1f2937 !important;
}
button[kind="primary"] {
    background: #3b82f6;
    color: #ffffff;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.75rem 1.5rem;
    border: none;
    transition: background 0.2s ease;
}
button[kind="primary"]:hover {
    background: #2563eb;
}
h1, h2, h3 {
    color: #1f2937;
    font-weight: 700;
}
.stSlider > div > div > div > div {
    color: #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_artifacts_or_raw():
    try:
        df = pd.read_parquet(os.path.join("artifacts", "outputs", "books_processed.parquet"))
        X = load_joblib("tfidf_matrix")
        labels = load_joblib("kmeans_labels")
        vec = None  # train.py does not save tfidf vectorizer
        eda_summary = load_json("eda_summary")
        nlp_summary = load_json("nlp_summary")

        genre_nonempty = df.get("genre", pd.Series([""] * len(df))).astype(str).str.strip().astype(bool).mean()
        year_nonnull = df.get("publication_year", pd.Series([np.nan] * len(df))).notna().mean()
        if genre_nonempty < 0.05 or year_nonnull < 0.05:
            raise RuntimeError("Artifacts missing critical data, rebuilding")

        return df.reset_index(drop=True), X, labels, vec, eda_summary, nlp_summary, "artifacts"
    except Exception as e:
        st.warning(f"Failed to load artifacts: {str(e)}. Rebuilding from raw data...")
        df1, df2 = load_raw(".")
        df = clean_and_merge(df1, df2)
        vec, X = fit_transform_tfidf(df)
        _, labels = fit_kmeans(X, n_clusters=20)
        eda_summary = {}  # Fallback if basic_eda is not run
        nlp_summary = {}  # Fallback if nlp_summary is not generated
        return df.reset_index(drop=True), X, labels, vec, eda_summary, nlp_summary, "raw"

def _genre_dummies(df):
    return df["genre"].fillna("").str.get_dummies(sep=",") if "genre" in df else pd.DataFrame(index=df.index)

def _find_exact_index(df, title, author=""):
    t, a = str(title).strip().lower(), str(author).strip().lower()
    cand = df[df.get("book_name", "").str.lower() == t]
    if not cand.empty and a:
        cand = cand[cand.get("author", "").str.lower() == a]
    return int(cand.index[0]) if not cand.empty else -1

def _style_nlp_similarity(df_view):
    if "nlp_similarity" not in df_view.columns:
        return df_view
    def color_cell(v):
        try:
            x = float(v)
            return "background-color: #d1fae5" if x >= 0.5 else "background-color: #fef3c7" if x >= 0.2 else "background-color: #ffedd5"
        except:
            return ""
    return df_view.style.applymap(color_cell, subset=["nlp_similarity"])

def show_about():
    st.subheader("About This App")
    st.markdown("""
    This book recommendation system merges two book datasets, extracts TF-IDF features, clusters books using KMeans, and provides personalized recommendations. Explore dataset insights, get tailored book suggestions, evaluate model performance, and analyze NLP-driven features through an intuitive interface.
    """)
    st.markdown("**Navigate using the sidebar**:")
    st.markdown("- **EDA**: Visualize genre popularity, ratings, and publication trends.")
    st.markdown("- **Recommendations**: Find books based on titles or preferences.")
    st.markdown("- **Insights**: Discover book clusters and key characteristics.")
    st.markdown("- **Evaluate**: Assess recommendation model accuracy.")
    st.markdown("- **NLP**: Explore top TF-IDF terms driving recommendations.")
    st.markdown("- **Conclusion**: Review findings and future improvements.")

def show_eda(df, eda_summary):
    st.subheader("Exploratory Data Analysis")
    st.write({"Books": df.shape[0], "Features": df.shape[1]})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Genres**")
        if "genre" in df:
            tg = _genre_dummies(df).sum().sort_values(ascending=False).head(10)
            if tg.sum() > 0:
                st.plotly_chart(px.bar(tg, title="Most Popular Genres", color_discrete_sequence=["#3b82f6"]), use_container_width=True)
            else:
                st.info("No genre data available.")
        else:
            st.info("No genre column detected.")
    with c2:
        st.markdown("**Rating Distribution**")
        if "rating" in df:
            st.plotly_chart(px.histogram(df, x="rating", nbins=30, title="Book Ratings", color_discrete_sequence=["#3b82f6"]), use_container_width=True)
        else:
            st.info("No rating data available.")

    st.markdown("**Publication Year Trends**")
    if "publication_year" in df and df["publication_year"].notna().sum() > 0:
        years = df["publication_year"].dropna().astype(int)
        year_counts = years.value_counts().sort_index()
        st.plotly_chart(px.line(x=year_counts.index, y=year_counts.values, title="Books Published Over Time", labels={"x": "Year", "y": "Number of Books"}, color_discrete_sequence=["#3b82f6"]), use_container_width=True)
    else:
        st.info("No publication year data available.")

    st.markdown("**EDA Summary (from training)**")
    if eda_summary:
        st.json(eda_summary)
    else:
        st.info("EDA summary not available. Run train.py to generate.")

def show_recommender(df, X, labels, vec):
    st.subheader("Personalized Book Recommendations")
    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.text_input("Enter a book title")
        author = st.text_input("Optional: Author name")
        query_text = st.text_area("Or describe your preferences (e.g., 'science fiction adventure')")
    with col2:
        top_k = st.slider("Number of recommendations", 5, 20, 10)
        method = st.radio("Recommendation Type", ["Hybrid", "Content-Based", "Cluster-Based"], index=0)

    def recommend_text(q, k=10):
        if not q.strip() or vec is None:
            st.warning("Text-based recommendations unavailable: TF-IDF vectorizer not loaded.")
            return pd.DataFrame(columns=df.columns)
        qv = vec.transform([q])
        sims = (qv @ X.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        recs = df.iloc[order].copy()
        recs["nlp_similarity"] = sims[order]
        return recs

    def recommend_hidden_gems(df, X, labels, k=5):
        low_popularity = df[df["number_of_reviews"] < df["number_of_reviews"].quantile(0.25)]
        high_rated = low_popularity[low_popularity["rating"] >= 4.0]
        if high_rated.empty:
            return pd.DataFrame(columns=df.columns)
        indices = high_rated.index[:k]
        recs = df.loc[indices].copy()
        recs["note"] = "Hidden Gem"
        return recs

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recs = pd.DataFrame()
            if title.strip():
                recs = (recommend_content_based(df, X, title, author, top_k) if method == "Content-Based"
                        else recommend_within_cluster(df, X, labels, title, author, top_k) if method == "Cluster-Based"
                        else recommend_hybrid(df, X, labels, title, author, top_k))
            recs_text = recommend_text(query_text, top_k)
            recs_gems = recommend_hidden_gems(df, X, labels, top_k) if query_text.lower().find("hidden gems") != -1 else pd.DataFrame()

            df_out = None
            if not recs.empty and not recs_text.empty:
                merged = recs.join(recs_text[["nlp_similarity"]], how="left", rsuffix="_text")
                merged["combined_score"] = merged.get("hybrid_score", merged.get("similarity", 0)).fillna(0) * 0.6 + merged["nlp_similarity"].fillna(0) * 0.4
                df_out = merged.sort_values("combined_score", ascending=False).head(top_k)
            elif not recs.empty:
                df_out = recs
            elif not recs_text.empty:
                df_out = recs_text
            if not recs_gems.empty:
                df_out = pd.concat([df_out, recs_gems], ignore_index=True) if df_out is not None else recs_gems

            if title.strip():
                idx = _find_exact_index(df, title, author)
                if idx != -1:
                    sel = df.loc[[idx]].copy()
                    sel["note"] = "Your Selection"
                    df_out = pd.concat([sel, df_out], ignore_index=True) if df_out is not None else sel

            if df_out is None or df_out.empty:
                st.info("No recommendations found. Try a different title or description.")
            else:
                display_cols = ["book_name", "author", "rating", "genre", "nlp_similarity"]
                if "note" in df_out.columns:
                    display_cols.append("note")
                st.dataframe(_style_nlp_similarity(df_out[display_cols].fillna("")), use_container_width=True)

def show_insights(df, X, labels, nlp_summary):
    st.subheader("Book Insights")
    if "cluster" not in df.columns:
        df["cluster"] = labels
    clusters = df.groupby("cluster").size().sort_values(ascending=False).head(10)
    rep = []
    for cid, size in clusters.items():
        group = df[df["cluster"] == cid]
        top = group.sort_values(["number_of_reviews", "rating"], ascending=[False, False]).head(3)
        top_authors = group["author"].value_counts().head(3).index.tolist()
        top_genres = _genre_dummies(group).sum().sort_values(ascending=False).head(3).index.tolist()
        rep_titles = nlp_summary.get("representative_titles_per_cluster", {}).get(str(cid), [])
        rep.append({
            "Cluster": int(cid),
            "Size": int(size),
            "Top Books": "; ".join(top["book_name"].fillna("").tolist()),
            "Representative Titles": "; ".join(rep_titles),
            "Top Authors": ", ".join(top_authors),
            "Top Genres": ", ".join(top_genres)
        })
    st.dataframe(pd.DataFrame(rep), use_container_width=True)

    st.markdown("**Top Authors by Average Rating**")
    author_ratings = df.groupby("author")["rating"].mean().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(x=author_ratings.values, y=author_ratings.index, orientation="h", title="Top Authors by Rating", labels={"x": "Average Rating", "y": "Author"}, color_discrete_sequence=["#3b82f6"]), use_container_width=True)

def show_evaluation(df, X, labels):
    st.subheader("Model Evaluation")
    if "genre" not in df:
        return st.info("Genre data unavailable for evaluation.")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of Samples", 20, 200, 60)
    with col2:
        k = st.slider("Recommendations per Sample (K)", 3, 20, 10)

    def compute(method):
        rng = np.random.default_rng(123)
        idxs = rng.choice(len(df), min(n, len(df)), replace=False)
        prec, rec, ndcg, cnt = 0, 0, 0, 0
        for idx in idxs:
            title = df.iloc[idx]["book_name"]
            recs = (recommend_content_based(df, X, title, None, k) if method == "content"
                    else recommend_within_cluster(df, X, labels, title, None, k) if method == "cluster"
                    else recommend_hybrid(df, X, labels, title, None, k))
            if recs.empty:
                continue
            qg = set(map(str.strip, str(df.iloc[idx].get("genre", "")).split(",")))
            rels = [(1 if qg & set(map(str.strip, str(r.get("genre", "")).split(","))) else 0) for _, r in recs.iterrows()]
            rels = np.array(rels, dtype=float)
            prec += rels.sum() / len(rels)
            rec += rels.sum() / len(qg) if qg else 0
            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rels), len(qg))))
            ndcg += dcg / idcg if idcg > 0 else 0
            cnt += 1
        return {
            f"Precision@{k}": prec / cnt if cnt else 0,
            f"Recall@{k}": rec / cnt if cnt else 0,
            f"NDCG@{k}": ndcg / cnt if cnt else 0
        }

    if st.button("Evaluate Models"):
        with st.spinner("Evaluating models..."):
            res = {
                "Content-Based": compute("content"),
                "Cluster-Based": compute("cluster"),
                "Hybrid": compute("hybrid")
            }
            df_res = pd.DataFrame(res).T
            st.dataframe(df_res.style.format("{:.3f}"), use_container_width=True)
            st.plotly_chart(px.bar(df_res, title="Model Performance Comparison", barmode="group", color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"]), use_container_width=True)

def show_nlp(df, labels, nlp_summary):
    st.subheader("NLP Insights (TF-IDF)")
    if not nlp_summary:
        return st.info("NLP summary not available. Run train.py to generate.")

    st.markdown("**Top Terms Across All Books**")
    overall = pd.DataFrame(nlp_summary.get("top_terms_overall", []))
    if not overall.empty:
        st.dataframe(overall.head(30), use_container_width=True)

    st.markdown("**Top Terms by Cluster**")
    for cid, terms in nlp_summary.get("top_terms_per_cluster", {}).items():
        with st.expander(f"Cluster {cid}"):
            df_terms = pd.DataFrame(terms)
            if not df_terms.empty:
                st.dataframe(df_terms.head(20), use_container_width=True)

def show_conclusion():
    st.subheader("Conclusion")
    st.markdown("""
    This system delivers robust book recommendations using NLP and clustering. Key findings:
    - **Hybrid Model**: Combines content and cluster-based approaches for optimal recommendations.
    - **Insights**: Popular genres (e.g., science fiction, thrillers) and high-rated authors drive engagement.
    - **Extensibility**: Ready to incorporate collaborative filtering or user history.
    - **Deployment**: Suitable for cloud hosting (e.g., AWS EC2).
    Future work could include real-time feedback and audiobook-specific features.
    """)

def main():
    st.title("Book Recommendation System")
    df, X, labels, vec, eda_summary, nlp_summary, source = load_artifacts_or_raw()
    st.caption(f"Data loaded from: {source}")

    PAGES = ["About", "EDA", "Recommendations", "Insights", "Evaluate", "NLP", "Conclusion"]
    icons = {"About": "🏠", "EDA": "📊", "Recommendations": "✨", "Insights": "🔎", "Evaluate": "🧪", "NLP": "🧠", "Conclusion": "✅"}

    if "active_page" not in st.session_state:
        st.session_state.active_page = "About"

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
        options = [f"{icons[p]} {p}" for p in PAGES]
        selection = st.radio(
            "",
            options,
            index=PAGES.index(st.session_state.active_page),
            label_visibility="collapsed",
            key="nav_radio"
        )
        selected_page = selection[2:]
        st.session_state.active_page = selected_page
        st.markdown('</div>', unsafe_allow_html=True)

    page = st.session_state.active_page
    if page == "About":
        show_about()
    elif page == "EDA":
        show_eda(df, eda_summary)
    elif page == "Recommendations":
        show_recommender(df, X, labels, vec)
    elif page == "Insights":
        show_insights(df, X, labels, nlp_summary)
    elif page == "Evaluate":
        show_evaluation(df, X, labels)
    elif page == "NLP":
        show_nlp(df, labels, nlp_summary)
    else:
        show_conclusion()

if __name__ == "__main__":
    main()