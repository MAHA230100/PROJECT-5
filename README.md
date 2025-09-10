# Book Recommendation System

A complete pipeline to prepare data, extract NLP features, cluster books, and serve recommendations via Streamlit. Includes content-based, cluster-based, and hybrid recommenders.

## Project Structure

```
PROJECT-5/
  Audible_Catlog.csv
  Audible_Catlog_Advanced_Features.csv
  app.py
  train.py
  requirements.txt
  src/
    __init__.py
    data.py
    features.py
    clustering.py
    recommenders.py
    utils.py
  artifacts/          # created after training
```

## Quickstart (Local)

1) Create venv and install deps:
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) Train and generate artifacts:
```
python train.py
```
Artifacts saved to `artifacts/` including:
- `artifacts/outputs/books_processed.parquet`
- `artifacts/models/tfidf.joblib`, `tfidf_matrix.joblib`, `kmeans_labels.joblib`
- `artifacts/outputs/eda_summary.json`

3) Run the app:
```
streamlit run app.py
```
Open the URL shown in the console.

## Features
- Data cleaning and merging from two CSVs
- TF-IDF features on combined text fields
- MiniBatch KMeans clustering
- Recommend by content similarity, within cluster, or hybrid
- EDA charts for genres, ratings vs reviews, and publication trends

## Evaluation
You can extend with offline evaluation (precision/recall@k) by preparing a small validation set of known similar pairs or by splitting titles and using nearest neighbors retrieval.

## Requirements Coverage

- **Datasets**: Uses `Audible_Catlog.csv` and `Audible_Catlog_Advanced_Features.csv`. Merging coalesces fields like `rating`, `number_of_reviews`, `price`, `genre`, `description`, `listening_time`, `ranks`.
- **EDA**:
  - **Popular genres**: Bar chart (Top 5) in EDA tab.
  - **Highest-rated authors**: Table of authors with avg rating (min 3 books).
  - **Rating distribution**: Histogram.
  - **Ratings vs reviews**: Correlation heatmap and scatter.
  - **Publication trends**: Line chart by `publication_year`.
- **Questions addressed**:
  - Easy: Popular genres, top-rated authors, rating distribution, publication trends, rating vs review-count trends (heatmap).
  - Medium: Clusters shown in Insights; genre-similarity impact measured; author popularity vs rating correlation; hybrid vs content vs cluster comparison.
  - Scenarios: Top 5 Sci‑Fi, similar for thriller lovers, hidden gems (high rating + low popularity) provided in Recommendations tab.
- **Modeling**:
  - Text features via TF‑IDF on `text_blob` (title + author + genre + description).
  - MiniBatchKMeans clustering with persisted labels.
  - Recommenders: content-based, within-cluster, and hybrid (content + cluster boost + rating/reviews).
- **Evaluation**:
  - In-app Evaluate tab computes **precision@k**, **recall@k**, **nDCG@k** using shared-genre as relevance proxy, and reports the best method.
  - Baseline **RMSE** for rating prediction (global mean and author-mean CV) for completeness.
- **Application**:
  - Streamlit interface with tabs: EDA, Recommendations, Insights, Evaluate.
  - Artifacts cached to `artifacts/` and auto-loaded in the app.
- **Deployment**:
  - Step-by-step EC2 guide to run Streamlit, security group note, and optional S3 artifact storage.

## How to Answer the Assignment Questions

- **Most popular genres**: EDA tab → Top 5 Genres bar chart.
- **Top authors by rating**: EDA tab → Top Authors table.
- **Average rating distribution**: EDA tab → Ratings Histogram.
- **Publication trends**: EDA tab → Publication Year Trend.
- **Ratings vs review counts**: EDA tab → Correlation heatmap.
- **Books clustered together**: Insights tab → Top clusters with examples.
- **Genre similarity effect**: Insights tab → Avg fraction of recommendations sharing genres.
- **Author popularity effect**: Insights tab → Popularity vs Avg Rating plot and correlation table.
- **Best feature combo**: Evaluate tab → Compare Content/Cluster/Hybrid (nDCG/Precision/Recall@k).
- **Sci‑Fi recommendations**: Recommendations tab → "Top 5 Science Fiction" button.
- **Thriller-lover recommendations**: Recommendations tab → "Similar to Thrillers Lovers" button.
- **Hidden gems**: Recommendations tab → "Hidden Gems" button.

## AWS Deployment (EC2)
1) Launch an EC2 instance (Ubuntu 22.04 t3.small or better) and allow inbound TCP 8501 in the security group.
2) SSH to the instance, install system deps:
```
sudo apt update && sudo apt install -y python3-venv python3-pip
```
3) Clone/upload this project to the instance. Then:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```
4) Run Streamlit on 0.0.0.0:
```
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
5) Visit `http://<EC2_PUBLIC_IP>:8501`.

(Optional) Use `tmux` or a systemd service to keep it running.

## S3 Storage (Optional)
- Upload `artifacts/` to an S3 bucket and download at app start with `boto3`.

## Notes
- If your CSVs contain different column names, the loader standardizes names and coalesces overlaps.
- For large datasets, consider reducing TF-IDF `max_features` and cluster count.

