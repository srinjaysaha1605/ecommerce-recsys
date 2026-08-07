# E-Commerce Customer Analytics & Recommendation System

A full-stack customer analytics platform combining segmentation, personalized product recommendations, churn prediction, and customer lifetime value (CLV) forecasting — built on real transaction data and deployed as an interactive dashboard.

**🔗 Live demo:** [ecom-customer-insights.streamlit.app](https://ecom-customer-insights.streamlit.app/)

---

## Overview

This project analyzes ~5,900 customers and 4,600+ products from the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset to answer four questions:

1. **Who are our customers?** → RFM-based segmentation (VIP, Regular, New, At Risk, Lost)
2. **What should we recommend to them?** → Collaborative filtering + content-based fallback
3. **Who's likely to churn?** → Binary classification on purchase behavior
4. **What's a customer worth?** → Lifetime value regression

All four feed into a single Streamlit dashboard.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Visualization | Plotly |
| Segmentation | Scikit-learn (KMeans) |
| Recommendations | `implicit` (ALS), Scikit-learn (TF-IDF) |
| Churn / CLV | XGBoost |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version control | GitHub |

---

## Methodology

### 1. Data Cleaning & EDA
Loaded and merged both sheets of the Online Retail II dataset, removed cancellations and null customer IDs, and engineered a `Revenue` column. Explored monthly revenue trends, top products, and revenue by country to validate data quality.

### 2. Feature Engineering (RFM)
Computed **Recency, Frequency, Monetary** per customer, plus average order value, unique products purchased, and customer tenure. Outliers (bulk-buyer accounts skewing the distribution) were capped at the 99th percentile and log-transformed before clustering, while original uncapped values were retained separately for accurate display.

### 3. Customer Segmentation
**KMeans (K=5)**, selected via elbow method and silhouette score (0.369) on scaled RFM features. Segments labeled by inspecting cluster centroids:

| Segment | Recency | Frequency | Monetary |
|---|---|---|---|
| VIP | 43 days | 19.1 orders | £8,940 |
| Regular | 65 days | 5.5 orders | £1,962 |
| New Customer | 100 days | 1.7 orders | £418 |
| At Risk | 398 days | 3.3 orders | £1,314 |
| Lost | 521 days | 1.2 orders | £249 |

### 4. Recommendation Engine
- **Popularity baseline** — cold-start fallback, most-purchased products.
- **Collaborative filtering** — Alternating Least Squares (`implicit` library) on the customer × product implicit-feedback matrix (purchase quantities). Achieved **Precision@10: 0.1875, NDCG@10: 0.1829** on a held-out test split — well above the typical 0.05–0.15 range for implicit-feedback retail data.
- **Content-based fallback** — TF-IDF similarity on product descriptions, used for new customers/products the ALS model hasn't seen.

### 5. Predictive Models
- **Churn (XGBoost Classifier):** predicts customers inactive 180+ days. ROC-AUC **0.932**, F1-score 0.81–0.88 across classes.
- **CLV (XGBoost Regressor):** predicts historic total spend from behavioral features. R² **0.969**, MAE £157.

### 6. Dashboard
Four-page Streamlit app:
- **Home** — key metrics at a glance
- **Segmentation** — interactive scatter plot + segment summary
- **Recommendations** — real-time ALS-powered product suggestions per customer
- **Churn & CLV** — risk and value table sorted by actual spend

---

## Project Structure

```
ecommerce-recsys/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── churn_model.pkl
│   ├── clv_model.pkl
│   ├── als_model.pkl
│   └── als_artifacts.pkl
└── data/
    ├── rfm_customers.csv
    └── product_catalog.csv
```

---

## Evaluation Metrics Summary

| Model | Metric | Score |
|---|---|---|
| Segmentation (KMeans) | Silhouette Score | 0.369 |
| Recommendations (ALS) | Precision@10 | 0.1875 |
| Recommendations (ALS) | NDCG@10 | 0.1829 |
| Churn (XGBoost) | ROC-AUC | 0.932 |
| CLV (XGBoost) | R² | 0.969 |
| CLV (XGBoost) | MAE | £157.13 |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — UK-based online retailer transactions, Dec 2009–Dec 2011.
