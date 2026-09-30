# E-Commerce Customer Analytics & Recommendation System

A machine-learning powered e-commerce analytics platform that uses historical transaction data to understand customer behavior, segment customers, recommend products, identify churn risk, and estimate customer monetary value.

**🔗 Live Demo:** [ecom-customer-insights.streamlit.app](https://ecom-customer-insights.streamlit.app/)

---

## What This Project Does

The system takes transaction-level retail data and builds four customer intelligence components:

| Problem | Approach | Output |
|---|---|---|
| Customer segmentation | RFM + KMeans | VIP, Regular, New, At Risk, Lost |
| Product recommendation | Implicit ALS | Personalized product rankings |
| Churn prediction | XGBoost Classifier | Churn risk |
| Customer value | XGBoost Regressor | Predicted monetary value |

These components are presented through a **Streamlit dashboard**.

---

## Dataset

The project uses the **UCI Online Retail II** dataset from a UK-based online retailer covering transactions from **December 2009 to December 2011**.

- ~5,900 customers
- 4,600+ products
- Transaction-level purchase data
- Customer IDs
- Product codes and descriptions
- Quantity, unit price and transaction dates

**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

---

## 1. Data Cleaning & Feature Engineering

The raw transaction data is cleaned before building the models.

### Cleaning

- Merge the available Online Retail II transaction sheets
- Remove cancelled transactions
- Remove records without a Customer ID
- Calculate transaction revenue

```text
Revenue = Quantity × UnitPrice
```

### Customer Features

Transactions are aggregated into customer-level behavioral features:

- **Recency** — days since last purchase
- **Frequency** — number of purchases/orders
- **Monetary** — total spending
- **Average Order Value**
- **Unique Products Purchased**
- **Customer Tenure**

Because monetary and frequency distributions contain extreme values, clustering features are capped at the **99th percentile** and log-transformed before scaling. Original uncapped values are retained separately for reporting.

---

# 2. Customer Segmentation

### Algorithm: KMeans

KMeans is used as an **unsupervised clustering algorithm** to group customers with similar purchasing behavior.

The input consists primarily of scaled RFM-based behavioral features.

### Selecting K

The number of clusters was evaluated using:

- Elbow method
- Silhouette score

The final configuration uses:

```text
K = 5
Silhouette Score = 0.369
```

The numerical clusters were then interpreted using their behavioral centroids.

| Segment | Recency | Frequency | Monetary |
|---|---:|---:|---:|
| VIP | 43 days | 19.1 orders | £8,940 |
| Regular | 65 days | 5.5 orders | £1,962 |
| New Customer | 100 days | 1.7 orders | £418 |
| At Risk | 398 days | 3.3 orders | £1,314 |
| Lost | 521 days | 1.2 orders | £249 |

> The segment names are business interpretations of the resulting KMeans clusters; they are not labels learned directly by KMeans.

---

# 3. Product Recommendation Engine

The recommendation system uses **implicit collaborative filtering with ALS (Alternating Least Squares)**.

### Interaction Data

A sparse:

```text
Customer × Product
```

matrix is created from purchase behavior, with purchase quantities used as implicit feedback.

The ALS model factorizes this interaction matrix into latent customer and product representations and uses those representations to rank products for each customer.

### Model

```text
Library: implicit
Algorithm: Alternating Least Squares (ALS)
Feedback: Purchase quantity
```

### Evaluation

The recommender was evaluated on a held-out test split:

| Metric | Score |
|---|---:|
| Precision@10 | 0.1875 |
| NDCG@10 | 0.1829 |

**Precision@10** measures the relevance of the recommended items within the top 10.

**NDCG@10** additionally considers the position of relevant items, giving greater importance to relevant products appearing higher in the ranking.

### Cold Start

The project also includes a **popularity-based fallback** for customers without sufficient interaction history.

A **TF-IDF vectorizer** is also included for content-based product similarity using product descriptions, providing an additional approach for customers/products that cannot be handled effectively by collaborative filtering.

---

# 4. Churn Prediction

### Algorithm: XGBoost Classifier

Churn is formulated as a binary classification problem.

For this project:

```text
No purchase for 180+ days → Churned
```

The model uses customer behavioral features to distinguish between churned and active customers.

### Evaluation

| Metric | Score |
|---|---:|
| ROC-AUC | 0.932 |
| F1-score | 0.81–0.88 |

**ROC-AUC** evaluates how well the classifier separates the two classes across classification thresholds.

**F1-score** balances precision and recall.

> The 180-day threshold is a project-specific definition of churn, not a universal business definition.

---

# 5. Customer Value / CLV Model

### Algorithm: XGBoost Regressor

The project models customer monetary value as a regression problem.

The current implementation predicts **historic total customer spend from behavioral features**.

### Evaluation

| Metric | Score |
|---|---:|
| R² | 0.969 |
| MAE | £157.13 |

- **R²** measures the proportion of variance explained by the regression model.
- **MAE** represents the average absolute difference between predicted and actual values.

> This implementation should be understood as historical customer-value regression rather than a complete future cash-flow CLV model.

---

# 6. Streamlit Dashboard

The trained artifacts and prepared datasets are loaded into a Streamlit application.

### Dashboard Pages

#### Home
Displays high-level customer metrics.

#### Segmentation
Interactive customer visualization using:

- Recency
- Monetary value
- Frequency
- Customer segment

#### Recommendations
Allows a customer to be selected and generates a ranked list of products using the saved ALS model.

Previously interacted products are filtered from the recommendation results.

#### Churn & CLV
Displays customer-level behavioral information, churn status and monetary value.

---

# 7. Models & Saved Artifacts

The project stores trained models and preprocessing artifacts so the dashboard does not need to retrain them every time it starts.

```text
models/
├── kmeans_model.pkl       # Trained KMeans segmentation model
├── scaler.pkl             # Feature scaling used for clustering
├── churn_model.pkl        # Trained XGBoost churn classifier
├── clv_model.pkl          # Trained XGBoost value regressor
├── als_model.pkl          # Trained implicit ALS recommender
└── als_artifacts.pkl      # ALS interaction matrix + customer/product mappings
```

Prepared datasets:

```text
data/
├── rfm_customers.csv      # Customer-level engineered features
└── product_catalog.csv    # Product codes and descriptions
```

---

# 8. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Customer Segmentation | Scikit-learn KMeans |
| Feature Scaling | Scikit-learn |
| Recommendation | `implicit` ALS |
| Content Similarity | Scikit-learn TF-IDF |
| Churn Prediction | XGBoost Classifier |
| Customer Value | XGBoost Regressor |
| Model Persistence | Joblib / Pickle |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

# 9. Evaluation Summary

| Component | Algorithm | Metric | Result |
|---|---|---|---:|
| Segmentation | KMeans | Silhouette | **0.369** |
| Recommendation | ALS | Precision@10 | **0.1875** |
| Recommendation | ALS | NDCG@10 | **0.1829** |
| Churn | XGBoost Classifier | ROC-AUC | **0.932** |
| Churn | XGBoost Classifier | F1 | **0.81–0.88** |
| Customer Value | XGBoost Regressor | R² | **0.969** |
| Customer Value | XGBoost Regressor | MAE | **£157.13** |

---

# 10. Project Structure

```text
ecommerce-recsys/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── churn_model.pkl
│   ├── clv_model.pkl
│   ├── als_model.pkl
│   └── als_artifacts.pkl
│
└── data/
    ├── rfm_customers.csv
    └── product_catalog.csv
```

---

# 11. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will then be available through the local Streamlit server.

---

## Key Takeaway

This project combines **customer segmentation, recommendation, classification, and regression** into one e-commerce analytics workflow:

```text
Transactions
     ↓
Customer Behavior
     ↓
┌──────────┬──────────────┬──────────────┐
│ KMeans   │ ALS          │ XGBoost      │
│          │              │              │
│ Segments │ Recommend    │ Churn / CLV  │
└──────────┴──────────────┴──────────────┘
     ↓
Actionable Customer Insights
```
