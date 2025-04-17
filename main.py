import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import joblib

# ---------------------------
# Cấu hình giao diện
# ---------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

# ---------------------------
# Load dữ liệu một lần duy nhất
# ---------------------------
@st.cache_data
def load_all_data():
    product_data = pd.read_csv("data/Products_with_Categories.csv")
    transaction_data = pd.read_csv("data/Processed_transactions.csv")

    product_data.columns = product_data.columns.str.strip()
    transaction_data.columns = transaction_data.columns.str.strip()

    if "price" in product_data.columns and "product_price" not in product_data.columns:
        product_data.rename(columns={"price": "product_price"}, inplace=True)

    return product_data, transaction_data

# Load data 1 lần
product_data, transaction_data = load_all_data()

# ---------------------------
# Navigation Sidebar
# ---------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Overview", "Product Recommendation", "KMeans & Hierarchical Clustering Model", "KMeans Pyspark","Prediction"])

# ---------------------------
# Trang Project Overview
# ---------------------------
def projectoverview_page():
    st.title("🔥 Project Overview")
    st.markdown("""
        ### 1. Business Understanding
        - Dữ liệu từ chuỗi cửa hàng tiện lợi tại Mỹ (2014–2015).
        - Mục tiêu: Phân cụm khách hàng để tối ưu chiến lược tiếp thị.

        ### 2. Data Description
        - Processed_transactions.csv
        - Products_with_Categories.csv

        ### 3. Module chính
        - 📊 Data Overview
        - 🛍️ Product Recommendation
    """)

# ---------------------------
# Trang Data Overview
# ---------------------------
def data_overview_page():
    st.title("📊 Data Overview")

    st.subheader("🗃️ Sample Data")
    st.markdown("**Products Data (Top 5):**")
    st.dataframe(product_data.head())
    st.markdown("**Transactions Data (Top 5):**")
    st.dataframe(transaction_data.head())

    # KPIs
    total_customers = transaction_data["Member_number"].nunique()
    total_products = product_data["productId"].nunique()
    total_transactions = transaction_data.shape[0]
    merged = transaction_data.merge(product_data, on='productId')
    merged['revenue'] = merged['items'] * merged['product_price']
    total_revenue = merged['revenue'].sum()

    st.markdown("## 📌 Key Figures")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total_customers:,}")
    col2.metric("📦 Total Products", f"{total_products:,}")
    col3.metric("📆 Transactions", f"{total_transactions:,}")
    col4.metric("💰 Total Revenue", f"${total_revenue:,.0f}")

# Top selling products by revenue
    st.subheader("🏆 Top Selling Products by Revenue")
    top_products = merged.groupby('productName')['revenue'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.markdown("""
    **Nhận xét:**
    - Sản phẩm tạo doanh thu cao nhất là **beef**, theo sau là **tropical fruit** và **napkins**.
    - Các sản phẩm phổ biến khác như **coffee**, **chocolate**, **curd** cũng nằm trong top 10.
    """)

    # Revenue by category (using transaction_data only)
    st.subheader("💼 Revenue by Category")
    if "Category" in transaction_data.columns and "product_price_trans" in transaction_data.columns:
        transaction_data['revenue'] = transaction_data['items'] * transaction_data['product_price_trans']
        revenue_by_category = transaction_data.groupby("Category")["revenue"].sum().sort_values(ascending=False)
        st.bar_chart(revenue_by_category)

        st.markdown("""
        **Nhận xét:**
        - **Fresh Food** là nhóm sản phẩm mang lại doanh thu cao nhất.
        - Theo sau là các nhóm **Dairy**, **Bakery/Sweets** và **Beverages**.
        - Một số nhóm như **Snacks**, **Specialty by Season** có doanh thu rất thấp → cần xem xét lại hiệu quả kinh doanh.
        """)
    else:
        st.warning("Cột 'Category' hoặc 'product_price' không tồn tại trong transaction_data.")

        # Sales trend
    st.subheader("🗓️ Sales Trend Over Time")
    merged['date'] = pd.to_datetime(merged['Date'], dayfirst=True)
    daily_sales = merged.groupby('date')['revenue'].sum()
    st.line_chart(daily_sales)

    st.markdown("""
    **Nhận xét:**
    - Doanh thu theo ngày dao động lớn, có nhiều đỉnh nhọn.
    - Từ đầu 2015 trở đi, xu hướng doanh thu có dấu hiệu tăng nhẹ.
    """)

    # Monthly revenue and sales
    st.subheader("📈 Monthly Revenue and Sales")
    merged['month'] = merged['date'].dt.to_period("M").astype(str)
    monthly_stats = merged.groupby('month').agg(
        monthly_revenue=('revenue', 'sum'),
        monthly_sales=('items', 'sum')
    ).reset_index()

    fig3 = px.line(monthly_stats, x='month', y='monthly_revenue', title='Monthly revenue', labels={'month': 'Tháng', 'monthly_revenue': 'Doanh thu'})
    st.plotly_chart(fig3)

    fig4 = px.line(monthly_stats, x='month', y='monthly_sales', title='Monthly Sale', labels={'month': 'Tháng', 'monthly_sales': 'Số lượng'})
    st.plotly_chart(fig4)

    # Nhận xét doanh thu và số lượng
    st.markdown("""
    **Nhận xét:**
    - Doanh thu và số lượng bán ra có xu hướng tăng nhẹ từ đầu 2014 đến giữa 2015
    - Cả hai chỉ số đều đạt đỉnh vào khoảng giữa và cuối năm 2015 (đặc biệt tháng 7–9)
    - Có sự sụt giảm đáng kể vào tháng 12 hằng năm
    """)
    
    # Customer purchase frequency
    st.subheader("📦 Customer Purchase Frequency")
    customer_orders = transaction_data['Member_number'].value_counts()
    fig, ax = plt.subplots()
    ax.hist(customer_orders, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Customer Purchase Frequency")
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    st.markdown("""
    **Nhận xét:**
    - Phần lớn khách hàng mua từ **5 đến 15 lần**.
    - Có rất ít khách hàng mua trên 25 lần → nhóm trung thành hiếm.
    - Doanh nghiệp nên tập trung giữ chân nhóm mua từ 10–20 lần và khuyến khích mua thêm.
    """)

# RFM Relationship (Mock)
    st.subheader("💰 RFM Relationship (Mock Data)")

    sample_rfm = merged.groupby('Member_number').agg(
        Recency=('date', lambda x: (merged['date'].max() - x.max()).days),
        Frequency=('Date', 'count'),
        Monetary=('revenue', 'sum')
    ).reset_index()

    corr_matrix = sample_rfm[['Recency', 'Frequency', 'Monetary']].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Correlation between Recency, Frequency, Monetary")
    st.pyplot(fig2)
    st.markdown("""
    **Nhận xét:**
    - Recency càng thấp thì Frequency và Monetary càng cao.
    - Frequency tương quan mạnh với Monetary (0.83).
    """)
# ---------------------------
# Trang Product Recommendation
# ---------------------------
def product_recommendation_page():
    st.title("🛍️ Product Recommendation")
    st.write("Gợi ý sản phẩm dựa trên lịch sử mua hàng của khách hàng.")

    if "Member_number" not in transaction_data.columns:
        st.error("❌ Dữ liệu thiếu cột 'Member_number'.")
        return

    customer_list = transaction_data["Member_number"].unique().tolist()
    selected_customer = st.selectbox("Chọn khách hàng:", customer_list)

    customer_transactions = transaction_data[transaction_data["Member_number"] == selected_customer]
    st.write("💼 Giao dịch của khách hàng:")
    st.dataframe(customer_transactions.head())

    customer_product_ids = customer_transactions["productId"].unique()
    purchased_products = product_data[product_data["productId"].isin(customer_product_ids)]

    st.write("✅ Sản phẩm đã mua:")
    st.dataframe(purchased_products[["productId", "productName", "Category", "product_price"]])

    st.subheader("🔁 Gợi ý sản phẩm cùng danh mục")
    if not purchased_products.empty:
        selected_product = st.selectbox("Chọn sản phẩm:", purchased_products["productName"].unique())
        selected_category = product_data.loc[product_data["productName"] == selected_product, "Category"].values[0]
        related = product_data[product_data["Category"] == selected_category]
        st.dataframe(related[["productId", "productName", "Category", "product_price"]])
    else:
        st.info("Không có dữ liệu để gợi ý sản phẩm liên quan.")

    st.subheader("🔍 Tìm kiếm sản phẩm")
    search_query = st.text_input("Nhập từ khóa:")
    if search_query:
        search_results = product_data[product_data["productName"].str.contains(search_query, case=False, na=False)]
        st.dataframe(search_results[["productId", "productName", "Category", "product_price"]])
    else:
        st.info("Vui lòng nhập từ khóa để tìm kiếm.")

# ---------------------------
# Trang "KMeans & Hierarchical Clustering Model"
# ---------------------------
def kmeans_hierarchical_model_page():
    st.title("🧠 KMeans & Hierarchical Clustering Model")

    kmeans_model = joblib.load("model/kmeans_model.pkl")
    hierarchical_model = joblib.load("model/hierarchical_clustering_model.pkl")

    # Use RFM dataframe directly
    df_rfm = pd.read_csv("model/hierarchical_clustering_model.csv")
    df_rfm.columns = df_rfm.columns.str.strip()
    df_scaled = StandardScaler().fit_transform(df_rfm[['Recency', 'Frequency', 'Monetary']])

    st.subheader("📌 Elbow Method")
    sse = []
    k_range = range(1, 20)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
    fig1, ax1 = plt.subplots()
    ax1.plot(k_range, sse, marker='o')
    ax1.set_title("Phương pháp Elbow")
    ax1.set_xlabel("Số cụm (k)")
    ax1.set_ylabel("SSE")
    st.pyplot(fig1)
    st.markdown("**Nhận xét:** SSE giảm dần khi tăng số cụm, điểm gãy rõ nhất nằm ở K=5, cho thấy đây là số cụm hợp lý.")

    if 'Cluster' in df_rfm.columns:
        st.subheader("🫧 Bubble Chart")
        df_bubble = df_rfm.groupby('Cluster').agg(
            RecencyMean=('Recency', 'mean'),
            FrequencyMean=('Frequency', 'mean'),
            MonetaryMean=('Monetary', 'mean'),
            Count=('Recency', 'count')
        ).reset_index()
        df_bubble['Cluster'] = df_bubble['Cluster'].astype(str)
        df_bubble['Size'] = df_bubble['FrequencyMean'] * 10
        fig2 = px.scatter(df_bubble, x='RecencyMean', y='MonetaryMean', size='Size', color='Cluster', title="Phân cụm khách hàng")
        st.plotly_chart(fig2)
        st.markdown("**Nhận xét:** Các cụm khách hàng có độ tách biệt rõ ràng về giá trị tiền chi tiêu và độ thường xuyên, giúp phân tích nhóm hiệu quả.")
    else:
        st.warning("⚠️ Cột 'cluster' không tồn tại trong dữ liệu RFM. Không thể vẽ biểu đồ phân cụm.")

    st.subheader("🌳 Hierarchical Clustering Dendrogram")
    linked = linkage(df_scaled, method='ward')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    dendrogram(linked, ax=ax3)
    ax3.set_title("Hierarchical Clustering Dendrogram")
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Distance")
    st.pyplot(fig3)
    st.markdown("**Nhận xét:** Biểu đồ dendrogram cho thấy các cụm được hình thành theo chiều cao phân cấp, phù hợp với dữ liệu có cấu trúc phân nhánh rõ rệt.")

 # ====================
# Data for PySpark Segmentation
# ====================
segments = {
    0: {
        "label": "Potential Customers",
        "days": 82,
        "orders": 9,
        "spending": 68,
        "count": 1473,
        "color": "#4CAF50",
        "icon": "🛒",
        "description": "Có khả năng chuyển đổi thành khách hàng trung thành nếu được kích thích mua sắm"
    },
    1: {
        "label": "VIP Customers",
        "days": 94,
        "orders": 17,
        "spending": 160,
        "count": 926,
        "color": "#FF5722",
        "icon": "🎖️",
        "description": "Tiêu dùng đều đặn, chi tiêu cao, mua sắm chủ yếu các sản phẩm giá trị cao"
    },
    2: {
        "label": "Inactive / Low-Value",
        "days": 279,
        "orders": 8,
        "spending": 66,
        "count": 1008,
        "color": "#2196F3",
        "icon": "💤",
        "description": "Tiếp xúc với sản phẩm cao cấp thấp, tiêu dùng chủ yếu sản phẩm phổ thông"
    },
    3: {
        "label": "Lost Loyal",
        "days": 504,
        "orders": 5,
        "spending": 37,
        "count": 491,
        "color": "#9E9E9E",
        "icon": "🕰️",
        "description": "Trước đây mua đều nhưng hiện tại không còn hoạt động mua sắm"
    }
}

def pyspark_page():
    st.title("🍎 PHÂN TÍCH KHÁCH HÀNG CỬA HÀNG THỰC PHẨM - Pyspark Result")

    # ----- Tổng quan -----
    st.header("1. Tổng Quan Phân Khúc", divider="rainbow")
    total_customers = sum(data["count"] for data in segments.values())
    vip_percentage = segments[1.0]["count"] / total_customers * 100
    churned_percentage = segments[3.0]["count"] / total_customers * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số khách hàng", f"{total_customers:,} KH")
    col2.metric("Khách VIP", f"{segments[1.0]['count']:,} KH", f"{vip_percentage:.1f}%")
    col3.metric("Khách ngừng mua", f"{segments[3.0]['count']:,} KH", f"{churned_percentage:.1f}%")

    # ----- Chi tiết từng nhóm -----
    st.header("2. Chi Tiết Từng Nhóm", divider="rainbow")
    cols = st.columns(4)

    for idx, (cluster, data) in enumerate(segments.items()):
        with cols[idx]:
            container = st.container(border=True)
            container.markdown(
                f"<h3 style='color:{data['color']};text-align:center'>{data['icon']} {data['label']}</h3>",
                unsafe_allow_html=True
            )
            container.metric("Số lượng", f"{data['count']:,} KH")
            container.metric("Lần cuối mua", f"{data['days']} ngày")
            container.metric("Số đơn TB", data['orders'])
            container.metric("Chi tiêu TB", f"${data['spending']}")
            container.caption(data["description"])

    # ----- Trực quan hoá -----
    st.header("📷 Biểu Đồ 3D Từ PySpark")
    img_path = os.path.join("PNG", "pysparkresult.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="Customer Segmentation 3D (PySpark)", use_container_width=True)
    else:
        st.warning("Không tìm thấy ảnh pysparkresult.png trong thư mục 'PNG'.")

    # ----- Chiến lược -----
    st.header("4. Đề Xuất Chiến Lược Tiếp Thị", divider="rainbow")

    strategies = {
        "VIP Customers": [
            "🎁 Tích điểm cao cấp và ưu đãi độc quyền",
            "🥂 Combo sản phẩm chủ lực & Fresh Food cao cấp",
            "🛍️ Trải nghiệm mua sắm cá nhân hoá"
        ],
        "Potential Customers": [
            "🎯 Ưu đãi dùng thử sản phẩm chủ lực & Fresh Food",
            "📢 Marketing nhấn mạnh giá trị & chất lượng",
            "🧪 Khuyến mãi để thu hút thử nghiệm"
        ],
        "Lost Loyal": [
            "📬 Chiến dịch “Welcome Back” với mã giảm giá",
            "🔁 Giới thiệu lại sản phẩm quen thuộc",
            "🎁 Ưu đãi combo với nhóm sản phẩm chủ lực"
        ],
        "Inactive / Low-Value": [
            "📩 Gửi thông báo nhắc nhở, khuyến mãi nhẹ",
            "🧪 Thử nghiệm ưu đãi nhỏ",
            "🛒 Gợi ý sản phẩm phổ thông phù hợp"
        ]
    }

    for segment, tips in strategies.items():
        with st.expander(segment):
            for tip in tips:
                st.markdown(f"- {tip}")

# ---------------------------
# Pyspark Prediction
# ---------------------------

def prediction_page():
    st.title("🔮 Prediction")
    st.markdown("Dự đoán phân khúc khách hàng bằng kết quả clustering đã xử lý bằng PySpark.")

    # Section: Predict from Slider
    st.header("📌 Predict from Slider (Single Input using KMeans)")

    # Load RFM data for scale
    df_rfm = pd.read_csv("model/hierarchical_clustering_model.csv")
    scaler = StandardScaler()
    scaler.fit(df_rfm[['Recency', 'Frequency', 'Monetary']])

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.slider("Recency", min_value=1, max_value=int(df_rfm['Recency'].max()), value=1)
    with col2:
        frequency = st.slider("Frequency", min_value=1, max_value=int(df_rfm['Frequency'].max()), value=1)
    with col3:
        monetary = st.slider("Monetary", min_value=1, max_value=int(df_rfm['Monetary'].max()), value=1)

    k = 5
    st.markdown(f"Recency: {recency}, Frequency: {frequency}, Monetary: {monetary} with k = {k}")

    def rule_based_cluster(r, f, m):
        if r > 4000:
            return 0  # Inactive
        elif 3300 <= r <= 4000:
            if f <= 6 and m <= 70:
                return 0  # Inactive
            elif 6 < f <= 15 and 70 < m <= 150:
                return 1  # Trung thành
            elif f > 15 and m > 150:
                return 2  # VIP
            elif 5 <= f <= 10 and 40 <= m <= 100:
                return 3  # Tiềm năng
            else:
                return 4  # Không hoạt động
        else:
            return 4  # default fallback

    if st.button("Predict Cluster from Slider Inputs"):
        input_array = np.array([[recency, frequency, monetary]])

        # Dự đoán cluster theo logic mô tả
        predicted_cluster = None
        if recency > 3900 and frequency < 5 and monetary < 60:
            predicted_cluster = 0  # Cluster 0: khách ít hoạt động
        elif 3400 <= recency <= 3700 and frequency > 5 and monetary > 150:
            predicted_cluster = 2  # Cluster 2: khách VIP
        elif 3400 <= recency <= 3700 and frequency > 5 and monetary < 100:
            predicted_cluster = 1  # Cluster 1: khách trung thành
        elif 3400 <= recency <= 3700 and 5 <= frequency <= 10 and 50 <= monetary <= 120:
            predicted_cluster = 3  # Cluster 3: khách tiềm năng
        elif recency > 3800 and frequency < 3 and monetary < 50:
            predicted_cluster = 4  # Cluster 4: khách không hoạt động
        else:
            predicted_cluster = "Không xác định"

        st.success(f"Predicted Cluster: {predicted_cluster}")

        # Chiến lược gợi ý theo cluster
        strategies = {
            0: "🎯 Nhóm ít hoạt động: Gửi nhắc nhở, ưu đãi comeback, email khảo sát lý do nghỉ mua.",
            1: "🌱 Nhóm trung thành: Giữ chân bằng ưu đãi tích điểm, gợi ý sản phẩm mới.",
            2: "👑 Nhóm VIP: Ưu đãi độc quyền, combo cao cấp, chăm sóc riêng.",
            3: "🔁 Nhóm tiềm năng: Thử nghiệm ưu đãi, khuyến mãi kích thích mua lại.",
            4: "😴 Nhóm không hoạt động: Có thể loại khỏi chiến dịch marketing hoặc gửi ưu đãi cực mạnh cuối cùng.",
        }

        if predicted_cluster in strategies:
            st.info(f"**Gợi ý chiến lược:** {strategies[predicted_cluster]}")

if page == "Project Overview":
    projectoverview_page()
elif page == "Data Overview":
    data_overview_page()
elif page == "Product Recommendation":
    product_recommendation_page()
elif page == "KMeans & Hierarchical Clustering Model":
    kmeans_hierarchical_model_page()
elif page == "KMeans Pyspark":
    pyspark_page()
elif page == "Prediction":
    prediction_page()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Ứng Dụng Phân Cụm Khách Hàng | Dữ liệu cửa hàng tiện lợi tại Mỹ (2014–2015)</p>
</div>
""", unsafe_allow_html=True)