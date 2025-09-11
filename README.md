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

## Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](./deployment/docs/DEPLOYMENT.md).

### Quick Start

1. **Initial Setup** (run once on EC2):
   ```bash
   curl -s https://raw.githubusercontent.com/yourusername/PROJECT-5/main/deployment/scripts/setup_server.sh | bash -s -- https://github.com/yourusername/PROJECT-5.git main
   ```

2. **Automated Deployments**:
   - Pushes to the `main` branch will automatically deploy to EC2
   - Configure GitHub Secrets:
     - `EC2_HOST`: Your EC2 public IP
     - `SSH_PRIVATE_KEY`: Your private key for EC2 access

3. **Access the application**:
   ```
   http://<EC2_PUBLIC_IP>:8501
   ```

### Manual Deployment

```bash
# On your EC2 instance:
cd /opt/app/repo
./deployment/scripts/deploy.sh main
```

## Notes
- If your CSVs contain different column names, the loader standardizes names and coalesces overlaps.
- For large datasets, consider reducing TF-IDF `max_features` and cluster count.

