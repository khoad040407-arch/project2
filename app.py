# app.py
import streamlit as st
import os
import json
import re
import PyPDF2
import docx
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ============= CONFIG =============
st.set_page_config(
    page_title="AI CV & JD ",
    page_icon="📄",
    layout="wide",
)

# ============= SETUP API =============
# Lấy API Key từ biến môi trường hoặc nhập trực tiếp từ giao diện
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Nếu chưa có trong biến môi trường, hiện ô nhập ở sidebar
    with st.sidebar:
        st.divider()
        api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        st.caption("Bạn có thể lấy key tại [aistudio.google.com](https://aistudio.google.com/)")

if api_key:
    genai.configure(api_key=api_key)

# ============= UTIL FUNCTIONS =============

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# ============= AI FUNCTIONS =============

def get_gemini_model(system_instruction=None, json_mode=False):
    """Cấu hình model Gemini"""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }
    
    # Bật chế độ JSON nếu cần
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    # Cấu hình an toàn để tránh bị chặn nội dung vô lý
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction=system_instruction,
        safety_settings=safety_settings
    )
    return model

def analyze_cv_jd(cv_text: str, jd_text: str, language: str = "vi"):
    if not api_key:
        st.error("Vui lòng nhập Gemini API Key để tiếp tục.")
        return None

    system_prompt = f"""
    Bạn là chuyên gia tuyển dụng nhân sự (HR) có kinh nghiệm.
    Nhiệm vụ: Phân tích sự phù hợp giữa CV ứng viên và Mô tả công việc (JD).
    
    Hãy trả về kết quả dưới dạng JSON (không dùng Markdown code block, chỉ trả về raw JSON) với cấu trúc sau:
    {{
      "match_score": int, // Thang điểm 0-100
      "seniority": "Intern/Entry | Junior | Mid | Senior | Lead/Manager", // Đánh giá trình độ dựa trên CV
      "summary_cv": "string", // Tóm tắt ngắn gọn CV (khoảng 2-3 câu)
      "summary_jd": "string", // Tóm tắt yêu cầu cốt lõi của JD (khoảng 2-3 câu)
      "strengths": ["..."], // Các điểm mạnh của ứng viên so với JD
      "gaps": ["..."], // Các kỹ năng/kinh nghiệm còn thiếu so với JD
      "recommended_keywords": ["..."], // Các từ khóa quan trọng trong JD mà CV đang thiếu
      "bullet_improvements": [ // Gợi ý viết lại 3 điểm quan trọng nhất trong CV để khớp JD hơn
          {{ "original": "...", "improved": "..." }}
      ]
    }}
    
    Ngôn ngữ phản hồi: {language} (Tiếng Việt hoặc English).
    """

    user_prompt = f"""
    === CV CỦA ỨNG VIÊN ===
    {cv_text}

    === MÔ TẢ CÔNG VIỆC (JD) ===
    {jd_text}
    """

    try:
        # Gọi Gemini với chế độ JSON
        model = get_gemini_model(system_instruction=system_prompt, json_mode=True)
        response = model.generate_content(user_prompt)
        
        # Parse JSON
        return json.loads(response.text)
        
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini API: {str(e)}")
        return None


def rewrite_section(cv_text: str, language: str = "vi"):
    if not api_key:
        st.error("Vui lòng nhập API Key.")
        return ""

    system_prompt = f"Bạn là chuyên gia viết CV chuyên nghiệp. Hãy viết lại nội dung người dùng cung cấp sao cho hấp dẫn, chuyên nghiệp, dùng từ ngữ hành động (action verbs), ngắn gọn súc tích. Ngôn ngữ: {language}."
    
    prompt = f"""
    Đoạn gốc cần viết lại:
    "{cv_text}"
    
    Hãy viết lại đoạn trên hay hơn:
    """

    try:
        model = get_gemini_model(system_instruction=system_prompt, json_mode=False)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# ============= UI SECTIONS =============

def render_header():
    st.markdown("## 📄 AI Resume & Job Match Assistant (Gemini Powered)")
    st.markdown(
        "Giúp bạn **đánh giá mức độ phù hợp giữa CV và JD**, "
        "phát hiện **khoảng trống kỹ năng** và **gợi ý chỉnh sửa** CV."
    )
    st.markdown("---")

def render_inputs():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ CV của bạn")
        cv_mode = st.radio(
            "Chọn cách nhập CV:",
            ["Upload file", "Dán text"],
            horizontal=True,
        )

        cv_text = ""
        cv_file = None

        if cv_mode == "Upload file":
            cv_file = st.file_uploader(
                "Upload CV (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="cv_file",
            )
            if cv_file is not None:
                try:
                    if cv_file.type == "application/pdf":
                        cv_text = extract_text_from_pdf(cv_file)
                    elif cv_file.type in [
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "application/msword"
                    ]:
                        cv_text = extract_text_from_docx(cv_file)
                    else:
                        cv_text = cv_file.read().decode("utf-8", errors="ignore")
                except Exception as e:
                    st.error(f"Không đọc được file: {e}")
        else:
            cv_text = st.text_area("Dán nội dung CV của bạn tại đây", height=300)

    with col2:
        st.subheader("2️⃣ Job Description (JD)")
        jd_text = st.text_area(
            "Dán JD / mô tả công việc",
            height=360,
            help="Copy JD từ website tuyển dụng hoặc mô tả do HR gửi."
        )

    return clean_text(cv_text), clean_text(jd_text)

def render_overview(analysis_result):
    if not analysis_result:
        return

    st.subheader("📊 Tổng quan mức độ phù hợp")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match score", f"{analysis_result.get('match_score', 0)} / 100")
    with col2:
        st.metric("Cấp độ phù hợp", analysis_result.get("seniority", "N/A"))
    with col3:
        st.metric("Số điểm mạnh", len(analysis_result.get("strengths", [])))

    with st.expander("Tóm tắt CV & JD"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Tóm tắt CV:**")
            st.write(analysis_result.get("summary_cv", ""))
        with col_b:
            st.markdown("**Tóm tắt JD:**")
            st.write(analysis_result.get("summary_jd", ""))

def render_details_tabs(analysis_result):
    if not analysis_result:
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["✅ Điểm mạnh", "⚠️ Khoảng trống", "🧩 Từ khóa gợi ý", "✏️ Ví dụ rewrite"]
    )

    with tab1:
        st.markdown("### ✅ Điểm mạnh so với JD")
        strengths = analysis_result.get("strengths", [])
        if strengths:
            for s in strengths:
                st.markdown(f"- {s}")
        else:
            st.info("Chưa phát hiện điểm mạnh cụ thể.")

    with tab2:
        st.markdown("### ⚠️ Khoảng trống / thiếu so với JD")
        gaps = analysis_result.get("gaps", [])
        if gaps:
            for g in gaps:
                st.markdown(f"- {g}")
        else:
            st.success("Không thấy khoảng trống đáng kể.")

    with tab3:
        st.markdown("### 🧩 Từ khóa & kỹ năng nên thêm vào CV")
        keywords = analysis_result.get("recommended_keywords", [])
        if keywords:
            st.write(", ".join(keywords))
        else:
            st.info("Không có gợi ý từ khóa thêm.")

    with tab4:
        st.markdown("### ✏️ Gợi ý rewrite các bullet/đoạn mô tả kinh nghiệm")
        bullets = analysis_result.get("bullet_improvements", [])
        if bullets:
            for item in bullets:
                with st.expander(f"📌 {item.get('original', '')[:60]}..."):
                    st.markdown("**Bản gốc:**")
                    st.write(item.get("original", ""))
                    st.markdown("**Phiên bản cải thiện:**")
                    st.write(item.get("improved", ""))
        else:
            st.info("AI chưa tạo ví dụ rewrite.")

def render_custom_rewrite(language: str):
    st.markdown("---")
    st.subheader("✨ Rewrite 1 đoạn CV cụ thể")

    text = st.text_area(
        "Dán 1 đoạn/bullet trong CV mà bạn muốn AI viết lại:",
        height=120,
    )
    if st.button("Rewrite đoạn này ✏️", use_container_width=True):
        if not text.strip():
            st.warning("Hãy nhập một đoạn text trước.")
        else:
            if not api_key:
                st.error("Vui lòng nhập API Key ở sidebar trước.")
            else:
                with st.spinner("Đang rewrite..."):
                    improved = rewrite_section(text, language=language)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Bản gốc:**")
                    st.write(text)
                with col2:
                    st.markdown("**Bản rewrite:**")
                    st.write(improved)

# ============= MAIN APP =============

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        language = st.selectbox(
            "Ngôn ngữ output",
            options=["vi", "en"],
            format_func=lambda x: "Tiếng Việt" if x == "vi" else "English",
        )
        st.caption("App sử dụng **Google Gemini API** (gemini-1.5-flash) để phân tích CV & JD.")

    # Header
    render_header()

    # Inputs
    cv_text, jd_text = render_inputs()

    st.markdown("---")
    analyze_btn = st.button("🚀 Phân tích CV & JD", type="primary", use_container_width=True)

    if analyze_btn:
        if not cv_text or not jd_text:
            st.error("Vui lòng nhập **cả CV và JD** trước khi phân tích.")
        else:
            if not api_key:
                st.error("⚠️ Chưa nhập API Key. Vui lòng nhập key vào thanh bên trái (Sidebar).")
            else:
                with st.spinner("Đang phân tích với Gemini AI..."):
                    analysis_result = analyze_cv_jd(cv_text, jd_text, language=language)

                # Hiển thị kết quả
                if analysis_result:
                    render_overview(analysis_result)
                    render_details_tabs(analysis_result)

    # Khu rewrite custom
    render_custom_rewrite(language)

if __name__ == "__main__":
    main()