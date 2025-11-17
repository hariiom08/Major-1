import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="E-commerce Sales Analysis",
    page_icon="🛍️",
    layout="wide"
)

# Load the saved model
model = joblib.load('random_forest_model.joblib')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('ecommerce_dataset_updated.csv')
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format="%d-%m-%Y")
    df['Quarter'] = df['Purchase_Date'].dt.quarter
    df['Month'] = df['Purchase_Date'].dt.month
    return df

df = load_data()

# Title
st.title("E-commerce Sales Analysis Dashboard 🛍️")

# Sidebar for prediction inputs
st.sidebar.header("Price Prediction")
price = st.sidebar.number_input('Enter Original Price (Rs.)', min_value=0.0, value=100.0)
discount = st.sidebar.slider('Select Discount %', 0, 100, 10)
month = st.sidebar.selectbox('Select Month', range(1, 13))
quarter = st.sidebar.selectbox('Select Quarter', range(1, 5))

# Calculate discount effect
discount_effect = (discount * price) / 100

# Create input data for prediction
input_data = pd.DataFrame([[price, discount, month, quarter, discount_effect]], 
                         columns=['Price (Rs.)', 'Discount (%)', 'Month', 'Quarter', 'Discount_effect'])

# Make prediction
if st.sidebar.button('Predict Price'):
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f'Predicted Final Price: Rs. {prediction:.2f}')

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sales Analysis", "Customer Segments", "Category Analysis", "More Visuals"])

with tab1:
    st.header("Monthly Sales and Revenue Trends")
    
    # Monthly trends
    monthly_sales = df.groupby('Month')['Product_ID'].count()
    monthly_revenue = df.groupby('Month')['Final_Price(Rs.)'].sum()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(monthly_sales, color='blue', marker='o', label='Sales')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Sales', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(monthly_revenue, color='red', marker='o', label='Revenue')
    ax2.set_ylabel('Revenue (Rs.)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Monthly Sales and Revenue Trends')
    st.pyplot(fig)
    plt.close()

with tab2:
    st.header("Customer Segmentation")

    # Customer segmentation (compact layout)
    def segment_consumer(spendings):
        if spendings < 100:
            return 'Low spender'
        if spendings < 300:
            return 'Medium spender'
        return 'High spender'

    # only create column if missing to avoid overwriting during reruns
    if 'User_segment' not in df.columns:
        df['User_segment'] = df['Final_Price(Rs.)'].apply(segment_consumer)

    # prepare counts
    segment_counts = df['User_segment'].value_counts()

    # place a smaller pie + compact bar side-by-side so it fits on one page
    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%',
               colors=['lightcoral', 'lightblue', 'lightgreen'], startangle=90)
        ax.axis('equal')
        plt.title('Customer Segments')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=segment_counts.values, y=segment_counts.index, palette='viridis', ax=ax)
        ax.set_xlabel('Number of Customers')
        ax.set_ylabel('')
        ax.set_title('Segment Counts')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("Category Analysis")
    
    # Category-wise sales
    category_sales = df.groupby('Category')['Product_ID'].count().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    category_sales.plot(kind='barh')
    plt.title('Number of Sales by Category')
    plt.xlabel('Number of Sales')
    st.pyplot(fig)
    plt.close()
    
    # Category revenue
    category_revenue = df.groupby('Category')['Final_Price(Rs.)'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    category_revenue.plot(kind='barh')
    plt.title('Revenue by Category')
    plt.xlabel('Revenue (Rs.)')
    st.pyplot(fig)
    plt.close()

# Additional insights
st.header("Key Insights")
col1, col2, col3 = st.columns(3)

with col1:
    total_revenue = df['Final_Price(Rs.)'].sum()
    st.metric("Total Revenue", f"Rs. {total_revenue:,.2f}")

with col2:
    total_sales = len(df)
    st.metric("Total Sales", f"{total_sales:,}")

with col3:
    avg_discount = df['Discount (%)'].mean()
    st.metric("Average Discount", f"{avg_discount:.1f}%")

with tab4:
    st.header("Additional Visualizations (Horizontal Scroll)")

    # helper: convert matplotlib fig to base64 image
    def fig_to_base64(fig, width=420):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{data}"

    imgs = []

    # 1) Top products by revenue
    top_products = df.groupby('Product_ID')['Final_Price(Rs.)'].sum().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6, 4))
    top_products.plot(kind='bar')
    plt.title('Top Products by Revenue (top 12)')
    plt.ylabel('Revenue (Rs.)')
    plt.xlabel('Product_ID')
    imgs.append(fig_to_base64(fig))
    plt.close()

    # 2) Discount distribution histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['Discount (%)'].dropna(), bins=20, kde=False, color='skyblue')
    plt.title('Discount % Distribution')
    plt.xlabel('Discount (%)')
    imgs.append(fig_to_base64(fig))
    plt.close()

    # 3) Category vs Discount_range heatmap (amount of sales)
    try:
        df['Discount_range'] = pd.cut(df['Discount (%)'], bins=[0,10,20,30,100], labels=['0-10%','10-20%','20-30%','30%+'], include_lowest=True)
        cat_disc = df.groupby(['Category','Discount_range'], observed=False)['Product_ID'].count().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cat_disc, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Sales count by Category & Discount Range')
        imgs.append(fig_to_base64(fig))
        plt.close()
    except Exception:
        pass

    # 4) Quarterly sales heatmap
    try:
        quarter_cat = df.groupby(['Quarter','Category'], observed=False)['Product_ID'].count().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(quarter_cat, annot=True, fmt='d', cmap='OrRd')
        plt.title('Quarterly Sales by Category')
        imgs.append(fig_to_base64(fig))
        plt.close()
    except Exception:
        pass

    # 5) Price vs Final Price scatter (sample)
    fig, ax = plt.subplots(figsize=(6, 4))
    sample = df.sample(n=min(500, len(df)), random_state=42)
    ax.scatter(sample['Price (Rs.)'], sample['Final_Price(Rs.)'], alpha=0.6)
    plt.title('Price vs Final Price (sample)')
    plt.xlabel('Price (Rs.)')
    plt.ylabel('Final_Price(Rs.)')
    imgs.append(fig_to_base64(fig))
    plt.close()

    # 6) Total sales over time (monthly)
    try:
        monthly_totals = df.groupby(pd.Grouper(key='Purchase_Date', freq='M'))['Product_ID'].count()
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly_totals.plot(kind='line', marker='o', ax=ax, color='tab:green')
        ax.set_title('Total Sales (Monthly)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        imgs.append(fig_to_base64(fig))
        plt.close()
    except Exception:
        pass

    # Render images inside a horizontal scrolling strip
    html = "<div style='display:flex;gap:16px;overflow-x:auto;padding:8px;'>"
    for img in imgs:
        html += f"<div style='flex:0 0 auto;width:420px;border-radius:6px;background:#ffffff;padding:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);'><img src='{img}' style='width:100%;height:auto;border-radius:4px;'/></div>"
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)