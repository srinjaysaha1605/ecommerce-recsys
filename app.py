import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Analytics", layout="wide")

@st.cache_data
def load_data():
    rfm = pd.read_csv('data/rfm_customers.csv')
    products = pd.read_csv('data/product_catalog.csv', dtype={'StockCode': str})
    products['StockCode'] = products['StockCode'].str.strip().str.strip("'\"")
    return rfm, products

import pickle

@st.cache_resource
def load_models():
    with open('models/als_artifacts.pkl', 'rb') as f:
        als_artifacts = pickle.load(f)
    return {
        'churn': joblib.load('models/churn_model.pkl'),
        'clv': joblib.load('models/clv_model.pkl'),
        'als': joblib.load('models/als_model.pkl'),
        'sparse_matrix': als_artifacts['sparse_matrix'],
        'customer_cat': als_artifacts['customer_cat'],
        'product_cat': als_artifacts['product_cat'],
    }

rfm, products = load_data()
models = load_models()

page = st.sidebar.radio("Navigate", ["Home", "Segmentation", "Recommendations", "Churn & CLV"])

if page == "Home":
    st.title("E-Commerce Customer Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(rfm):,}")
    col2.metric("Avg Order Value", f"£{rfm['AvgOrderValue'].mean():.2f}")
    col3.metric("Churn Rate", f"{rfm['Churned'].mean()*100:.1f}%")

elif page == "Segmentation":
    st.title("Customer Segments")
    fig = px.scatter(rfm, x='Recency', y='Monetary', color='Segment', size='Frequency',
                      hover_data=['CustomerID'], render_mode='svg')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1))

elif page == "Recommendations":
    st.title("Product Recommendations")
    customer_id = st.selectbox("Select Customer", rfm['CustomerID'].unique())

    customer_cat = models['customer_cat']
    product_cat = models['product_cat']
    sparse_matrix = models['sparse_matrix']
    als_model = models['als']

    if customer_id in customer_cat:
        idx = customer_cat.get_loc(customer_id)
        ids, scores = als_model.recommend(idx, sparse_matrix[idx], N=10, filter_already_liked_items=True)
        product_codes = [str(c) for c in product_cat[ids]]  # force plain python str, no numpy/category quirks
        result = pd.DataFrame({'StockCode': product_codes, 'Score': scores})
        result['StockCode'] = result['StockCode'].astype(str)
        products['StockCode'] = products['StockCode'].astype(str)
        result = result.merge(products, on='StockCode', how='left')
        st.dataframe(result[['StockCode', 'Description', 'Score']])
    else:
        st.write("New customer — showing popular products instead")
        st.dataframe(products.head(10))

elif page == "Churn & CLV":
    st.title("Churn Risk & Lifetime Value")
    st.dataframe(rfm[['CustomerID','Segment','Recency','Monetary_actual','Churned']].sort_values('Monetary_actual', ascending=False))
