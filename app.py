import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import time

# --- CẤU HÌNH TRANG (Phải đặt đầu tiên) ---
st.set_page_config(
    page_title="Sentimind - AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ---

# 1. Load Model AI (Cache để không load lại nhiều lần)
@st.cache_resource
def load_sentiment_model():
    # Sử dụng model DistilBERT được fine-tune cho phân tích cảm xúc
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Load Animation Lottie từ URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- KHỞI TẠO ---
sentiment_pipeline = load_sentiment_model()
lottie_ai_robot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_analyzing = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_w51pcehl.json")

# --- CSS TÙY CHỈNH (Để ẩn menu mặc định và footer cho chuyên nghiệp hơn) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (THANH ĐIỀU HƯỚNG) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.markdown("## **Sentimind AI**")
    
    # Menu điều hướng xịn xò
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Live Analysis", "Batch Processing", "About Team"],
        icons=["speedometer2", "cpu", "cloud-upload", "people"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#ffffff"},
            "icon": {"color": "#4e73df", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4e73df"},
        }
    )
    st.info("Project 2 - Introduction to IT\n\n© 2025 Group Name")

# --- NỘI DUNG CHÍNH ---

# TAB 1: DASHBOARD (Tổng quan)
if selected == "Dashboard":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("📊 Business Insight Dashboard")
        st.markdown("Chào mừng quay trở lại! Dưới đây là tổng quan về cảm xúc khách hàng trong tháng này.")
    with col2:
        st_lottie(lottie_analyzing, height=150, key="dashboard_anim")

    st.markdown("---")
    
    # KPIs (Thẻ số liệu)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Tổng phản hồi", value="1,500", delta="120 review mới")
    with kpi2:
        st.metric(label="Tích cực (Positive)", value="1,100", delta="15%", delta_color="normal")
    with kpi3:
        st.metric(label="Tiêu cực (Negative)", value="400", delta="-5%", delta_color="inverse")
    with kpi4:
        st.metric(label="Độ tin cậy AI", value="98.5%", delta="Ổn định")

    # Biểu đồ mẫu (Giả lập dữ liệu để Demo Dashboard)
    st.subheader("📈 Xu hướng cảm xúc theo thời gian")
    chart_data = pd.DataFrame({
        'Ngày': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Positive': [100, 120, 115, 130, 150, 180, 190],
        'Negative': [20, 15, 25, 10, 20, 30, 25]
    })
    
    # Vẽ biểu đồ Line chart
    fig = px.line(chart_data, x='Ngày', y=['Positive', 'Negative'], 
                  labels={'value': 'Số lượng review', 'variable': 'Loại cảm xúc'},
                  color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: LIVE ANALYSIS (Demo trực tiếp)
elif selected == "Live Analysis":
    st.title("🧠 AI Phân Tích Trực Tiếp")
    st.write("Nhập bất kỳ câu phản hồi nào (Tiếng Anh) để xem AI phân tích thời gian thực.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        user_text = st.text_area("Nhập nội dung tại đây:", height=200, 
                                 placeholder="Ví dụ: I absolutely love this product! The quality is amazing.")
        analyze_btn = st.button("🚀 Phân tích ngay", type="primary")

    with col_result:
        if analyze_btn and user_text:
            with st.spinner("AI đang suy nghĩ..."):
                time.sleep(1) # Giả lập độ trễ một chút cho hiệu ứng
                result = sentiment_pipeline(user_text)[0]
                label = result['label']
                score = result['score']
                
                # Hiển thị kết quả dựa trên nhãn
                if label == 'POSITIVE':
                    st.success(f"### Kết quả: TÍCH CỰC (Positive) 😊")
                    st_lottie("https://assets10.lottiefiles.com/packages/lf20_5tjfcwda.json", height=150, key="happy")
                else:
                    st.error(f"### Kết quả: TIÊU CỰC (Negative) 😞")
                    st_lottie("https://assets10.lottiefiles.com/packages/lf20_kcxmcc.json", height=150, key="sad")
                
                st.progress(score, text=f"Độ tin cậy của AI: {score:.4f}")
        elif analyze_btn and not user_text:
            st.warning("Vui lòng nhập nội dung trước khi bấm nút.")
        else:
            st.info("Kết quả sẽ hiển thị tại đây...")
            st_lottie(lottie_ai_robot, height=200, key="waiting_robot")

# TAB 3: BATCH PROCESSING (Xử lý file)
elif selected == "Batch Processing":
    st.title("📂 Phân Tích Hàng Loạt")
    st.write("Tải lên file Excel/CSV chứa danh sách feedback để phân tích tự động.")
    
    uploaded_file = st.file_uploader("Chọn file dữ liệu", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Đọc file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Dữ liệu gốc (5 dòng đầu):")
            st.dataframe(df.head())
            
            # Chọn cột để phân tích
            text_column = st.selectbox("Chọn cột chứa nội dung feedback:", df.columns)
            
            if st.button("⚡ Chạy AI cho toàn bộ file"):
                with st.spinner("Đang xử lý dữ liệu lớn... Vui lòng đợi"):
                    # Chạy model cho từng dòng (Lưu ý: Demo chỉ chạy 10 dòng đầu để nhanh)
                    # Thực tế có thể bỏ .head(10) đi
                    results = []
                    for text in df[text_column].astype(str).head(20): 
                        res = sentiment_pipeline(text[:512])[0] # Cắt chuỗi nếu quá dài
                        results.append(res['label'])
                    
                    # Gán kết quả vào DataFrame (cho 20 dòng đầu demo)
                    df_result = df.head(20).copy()
                    df_result['AI Prediction'] = results
                    
                    st.success("Đã phân tích xong 20 dòng đầu tiên!")
                    st.dataframe(df_result)
                    
                    # Vẽ biểu đồ tổng kết
                    fig_pie = px.pie(df_result, names='AI Prediction', title='Tỷ lệ cảm xúc trong file', 
                                     color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"})
                    st.plotly_chart(fig_pie)
                    
        except Exception as e:
            st.error(f"Có lỗi khi đọc file: {e}")

# TAB 4: ABOUT TEAM
elif selected == "About Team":
    st.title("👋 Giới thiệu Nhóm")
    st.markdown("""
    ### Project 2: Developing an AI Application
    **Môn học:** Introduction to Information Technology  
    **Giảng viên:** [Tên Giảng Viên]
    
    ---
    ### Thành viên nhóm:
    1. **Nguyễn Văn A** - *Team Leader & Backend Dev*
    2. **Trần Thị B** - *Frontend Dev & UI/UX*
    3. **Lê Văn C** - *Data Engineer*
    4. **Phạm Thị D** - *Report & Presentation*
    
    ---
    ### Công nghệ sử dụng:
    * **Python & Streamlit:** Xây dựng ứng dụng Web.
    * **Hugging Face Transformers:** Mô hình AI (DistilBERT).
    * **Pandas & Plotly:** Xử lý và trực quan hóa dữ liệu.
    """)
    st.balloons() # Hiệu ứng bóng bay chào mừng