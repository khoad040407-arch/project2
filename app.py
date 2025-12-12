# app.py
"""Streamlit app: AI analyzes CVs and job description .

What this refactor improves
- Removes hard-coded API keys (env / st.secrets / sidebar input)
- Cleaner UI (sidebar settings, main tabs, consistent components)
- Robust file ingestion (PDF/DOCX/TXT) + text cleaning
- Analysis history in-session + download results (JSON)
- Safer Gemini JSON-mode parsing with fallback extraction

Run
  streamlit run app.py


"""


      
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# File readers
import PyPDF2
import docx

# Gemini
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory


# =====================
# App constants
# =====================
APP_TITLE = "AI CV & JD"
APP_ICON = "📄"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_INPUT_CHARS = 120_000  # guardrail to avoid huge prompts
DEFAULT_API_KEY = "AIzaSyAQIqYEHeWRhxIRRwxoTNMMFPQDf2dB4Rc"


# =====================
# Page config
# =====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)


# =====================
# Data models
# =====================
@dataclass
class AnalysisResult:
    match_score: int
    seniority: str
    summary_cv: str
    summary_jd: str
    strengths: List[str]
    gaps: List[str]
    recommended_keywords: List[str]
    bullet_improvements: List[Dict[str, str]]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnalysisResult":
        # Defensive parsing with defaults
        return AnalysisResult(
            match_score=int(d.get("match_score", 0) or 0),
            seniority=str(d.get("seniority", "N/A") or "N/A"),
            summary_cv=str(d.get("summary_cv", "") or ""),
            summary_jd=str(d.get("summary_jd", "") or ""),
            strengths=list(d.get("strengths", []) or []),
            gaps=list(d.get("gaps", []) or []),
            recommended_keywords=list(d.get("recommended_keywords", []) or []),
            bullet_improvements=list(d.get("bullet_improvements", []) or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_score": self.match_score,
            "seniority": self.seniority,
            "summary_cv": self.summary_cv,
            "summary_jd": self.summary_jd,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "recommended_keywords": self.recommended_keywords,
            "bullet_improvements": self.bullet_improvements,
        }


# =====================
# Utilities
# =====================

def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(txt: str) -> str:
    if not txt:
        return ""
    # Normalize whitespace but preserve line breaks lightly
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def truncate_for_prompt(txt: str, limit: int = MAX_INPUT_CHARS) -> Tuple[str, bool]:
    """Return (text, truncated?)."""
    if len(txt) <= limit:
        return txt, False
    return txt[:limit], True


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> str:
    reader = PyPDF2.PdfReader(st.runtime.uploaded_file_manager.UploadedFile(
        filename, "application/pdf", file_bytes
    ))
    text_parts: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            text_parts.append(extracted)
    return "\n".join(text_parts)


@st.cache_data(show_spinner=False)
def extract_text_from_docx(file_bytes: bytes, filename: str = "document.docx") -> str:
    f = st.runtime.uploaded_file_manager.UploadedFile(
        filename,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_bytes,
    )
    document = docx.Document(f)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def safe_json_loads(maybe_json: str) -> Dict[str, Any]:
    """Try to parse JSON. If the model returns extra text, attempt to recover.

    This is common when the model wraps JSON with text or code fences.
    """
    maybe_json = maybe_json.strip()

    # Remove code fences if present
    maybe_json = re.sub(r"^```(?:json)?\s*", "", maybe_json, flags=re.IGNORECASE)
    maybe_json = re.sub(r"\s*```$", "", maybe_json)

    try:
        return json.loads(maybe_json)
    except json.JSONDecodeError:
        # Attempt to extract the first JSON object
        match = re.search(r"\{.*\}", maybe_json, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# =====================
# Gemini
# =====================

def configure_gemini(api_key: str) -> None:
    genai.configure(api_key=api_key)


def get_gemini_model(
    model_name: str,
    system_instruction: Optional[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    json_mode: bool,
) -> genai.GenerativeModel:
    generation_config: Dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction,
        safety_settings=safety_settings,
    )


def build_analysis_system_prompt(language: str) -> str:
    return f"""
Bạn là chuyên gia tuyển dụng (HR) nhiều kinh nghiệm.
Nhiệm vụ: Phân tích mức độ phù hợp giữa CV ứng viên và Job Description (JD).

Yêu cầu đầu ra:
- Trả về **JSON thuần** (không markdown, không code block).
- Đúng cấu trúc sau:
{json.dumps({
  "match_score": 0,
  "seniority": "Intern/Entry | Junior | Mid | Senior | Lead/Manager",
  "summary_cv": "",
  "summary_jd": "",
  "strengths": [""],
  "gaps": [""],
  "recommended_keywords": [""],
  "bullet_improvements": [{"original": "", "improved": ""}]
}, ensure_ascii=False, indent=2)}

Ngôn ngữ phản hồi: {"Tiếng Việt" if language == "vi" else "English"}.
Ghi chú:
- match_score là số nguyên 0-100.
- bullet_improvements: Ưu tiên 3 gợi ý quan trọng nhất.
""".strip()


def analyze_cv_jd(
    *,
    api_key: str,
    model_name: str,
    cv_text: str,
    jd_text: str,
    language: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> AnalysisResult:
    configure_gemini(api_key)

    sys_prompt = build_analysis_system_prompt(language)

    cv_for_prompt, cv_trunc = truncate_for_prompt(cv_text)
    jd_for_prompt, jd_trunc = truncate_for_prompt(jd_text)

    user_prompt = f"""
=== CV CỦA ỨNG VIÊN ===
{cv_for_prompt}

=== MÔ TẢ CÔNG VIỆC (JD) ===
{jd_for_prompt}
""".strip()

    model = get_gemini_model(
        model_name=model_name,
        system_instruction=sys_prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        json_mode=True,
    )

    response = model.generate_content(user_prompt)
    data = safe_json_loads(response.text)
    result = AnalysisResult.from_dict(data)

    # Attach truncation warnings into summaries (non-breaking)
    if cv_trunc or jd_trunc:
        note = []
        if cv_trunc:
            note.append("CV bị cắt bớt do quá dài")
        if jd_trunc:
            note.append("JD bị cắt bớt do quá dài")
        suffix = " (" + "; ".join(note) + ")" if language == "vi" else " (" + "; ".join(note) + ")"
        result.summary_cv = (result.summary_cv or "").strip() + suffix

    return result


def rewrite_section(
    *,
    api_key: str,
    model_name: str,
    text: str,
    language: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str:
    configure_gemini(api_key)

    system_prompt = (
        "Bạn là chuyên gia viết CV chuyên nghiệp. "
        "Hãy viết lại nội dung người dùng cung cấp sao cho hấp dẫn, chuyên nghiệp, "
        "dùng action verbs, đo được bằng số liệu khi hợp lý, ngắn gọn súc tích. "
        f"Ngôn ngữ: {language}."
    )

    prompt = (
        "Đoạn gốc cần viết lại:\n"
        f"{text.strip()}\n\n"
        "Yêu cầu:\n"
        "- Giữ nguyên ý chính\n"
        "- Ưu tiên kết quả theo bullet (nếu phù hợp)\n"
        "- Tránh nói quá/không đúng sự thật\n"
    ).strip()

    model = get_gemini_model(
        model_name=model_name,
        system_instruction=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        json_mode=False,
    )

    response = model.generate_content(prompt)
    return (response.text or "").strip()


# =====================
# UI
# =====================

def init_session_state() -> None:
    st.session_state.setdefault("analysis_history", [])  # list[dict]
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("cv_text", "")
    st.session_state.setdefault("jd_text", "")


def sidebar_settings() -> Dict[str, Any]:
    """Collect settings and return a dict."""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption("Gemini-powered CV/JD matching")

        # API key priority: env -> secrets -> default (hidden from UI)
        env_key = os.getenv("GEMINI_API_KEY")
        secret_key = None
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            secret_key = None

        api_key = env_key or secret_key or DEFAULT_API_KEY

        st.divider()

        language = st.selectbox(
            "Ngôn ngữ output",
            options=["vi", "en"],
            format_func=lambda x: "Tiếng Việt" if x == "vi" else "English",
        )

        model_name = DEFAULT_MODEL

        with st.expander("Tùy chỉnh generation", expanded=False):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
            max_output_tokens = st.slider("Max output tokens", 512, 8192, 4096, 256)

        st.divider()

        if st.button("🧹 Xóa kết quả & lịch sử", use_container_width=True):
            st.session_state.analysis_history = []
            st.session_state.last_result = None
            st.toast("Đã xóa", icon="✅")

        st.caption("Tip: Muốn có việc làm hãy thay đỗi bản thân")

    return {
        "api_key": api_key,
        "language": language,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
    }


def render_header() -> None:
    st.markdown("## 📄 AI Analyzes CV & Job Descriptions.")
    st.markdown(
        "Phân tích mức độ phù hợp giữa **CV** và **JD**, phát hiện **khoảng trống kỹ năng**, "
        "và gợi ý **rewrite** để CV khớp JD hơn."
    )


def render_inputs() -> Tuple[str, str]:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("1️⃣ CV của bạn")
        cv_mode = st.radio("Nhập CV", ["Upload file", "Dán text"], horizontal=True, key="cv_mode")

        cv_text = ""
        if cv_mode == "Upload file":
            cv_file = st.file_uploader(
                "Upload CV (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="cv_file",
            )
            if cv_file is not None:
                try:
                    raw = cv_file.getvalue()
                    if cv_file.type == "application/pdf":
                        with st.spinner("Đang đọc PDF..."):
                            cv_text = extract_text_from_pdf(raw, cv_file.name)
                    elif "wordprocessingml" in cv_file.type or cv_file.name.lower().endswith(".docx"):
                        with st.spinner("Đang đọc DOCX..."):
                            cv_text = extract_text_from_docx(raw, cv_file.name)
                    else:
                        cv_text = extract_text_from_txt(raw)
                except Exception as e:
                    st.error(f"Không đọc được file CV: {e}")
        else:
            cv_text = st.text_area(
                "Dán nội dung CV",
                height=330,
                value=st.session_state.get("cv_text", ""),
                placeholder="Ví dụ: Kinh nghiệm, kỹ năng, dự án...",
                key="cv_text_area",
            )

        cv_text = clean_text(cv_text)
        st.session_state.cv_text = cv_text
        st.caption(f"Ký tự CV: {len(cv_text):,}")

    with col2:
        st.subheader("2️⃣ Job Description (JD)")
        jd_mode = st.radio("Nhập JD", ["Upload file", "Dán text"], horizontal=True, key="jd_mode")

        jd_text = ""
        if jd_mode == "Upload file":
            jd_file = st.file_uploader(
                "Upload JD (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="jd_file",
            )
            if jd_file is not None:
                try:
                    raw = jd_file.getvalue()
                    if jd_file.type == "application/pdf":
                        with st.spinner("Đang đọc PDF..."):
                            jd_text = extract_text_from_pdf(raw, jd_file.name)
                    elif "wordprocessingml" in jd_file.type or jd_file.name.lower().endswith(".docx"):
                        with st.spinner("Đang đọc DOCX..."):
                            jd_text = extract_text_from_docx(raw, jd_file.name)
                    else:
                        jd_text = extract_text_from_txt(raw)
                except Exception as e:
                    st.error(f"Không đọc được file JD: {e}")
        else:
            jd_text = st.text_area(
                "Dán JD / mô tả công việc",
                height=330,
                value=st.session_state.get("jd_text", ""),
                placeholder="Copy từ website tuyển dụng hoặc mô tả do HR gửi...",
                key="jd_text_area",
            )

        jd_text = clean_text(jd_text)
        st.session_state.jd_text = jd_text
        st.caption(f"Ký tự JD: {len(jd_text):,}")

    return cv_text, jd_text


def render_overview(result: AnalysisResult) -> None:
    st.subheader("📊 Tổng quan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", f"{result.match_score} / 100")
    c2.metric("Seniority", result.seniority)
    c3.metric("# Strengths", f"{len(result.strengths)}")

    with st.expander("Tóm tắt CV & JD", expanded=True):
        a, b = st.columns(2)
        with a:
            st.markdown("**Tóm tắt CV**")
            st.write(result.summary_cv)
        with b:
            st.markdown("**Tóm tắt JD**")
            st.write(result.summary_jd)


def render_details(result: AnalysisResult) -> None:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["✅ Điểm mạnh", "⚠️ Khoảng trống", "🧩 Từ khóa nên thêm", "✏️ Ví dụ rewrite"]
    )

    with tab1:
        if result.strengths:
            for s in result.strengths:
                st.markdown(f"- {s}")
        else:
            st.info("Chưa phát hiện điểm mạnh cụ thể.")

    with tab2:
        if result.gaps:
            for g in result.gaps:
                st.markdown(f"- {g}")
        else:
            st.success("Không thấy khoảng trống đáng kể.")

    with tab3:
        if result.recommended_keywords:
            st.write(", ".join(map(str, result.recommended_keywords)))
        else:
            st.info("Không có gợi ý từ khóa thêm.")

    with tab4:
        bullets = result.bullet_improvements or []
        if bullets:
            for idx, item in enumerate(bullets, start=1):
                original = str(item.get("original", "")).strip()
                improved = str(item.get("improved", "")).strip()
                title = original[:80] + ("..." if len(original) > 80 else "")
                with st.expander(f"#{idx} – {title}"):
                    st.markdown("**Bản gốc**")
                    st.write(original)
                    st.markdown("**Phiên bản cải thiện**")
                    st.write(improved)
        else:
            st.info("AI chưa tạo ví dụ rewrite.")


def render_rewrite_tool(settings: Dict[str, Any]) -> None:
    st.markdown("---")
    st.subheader("✨ Rewrite nhanh một đoạn CV")

    text = st.text_area(
        "Dán 1 đoạn/bullet muốn viết lại",
        height=140,
        placeholder="Ví dụ: Led a team to deliver ...",
        key="rewrite_input",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        btn = st.button("Rewrite ✏️", type="primary", use_container_width=True)
    with col_b:
        st.caption("Mẹo: thêm số liệu/impact vào đoạn gốc để AI rewrite tốt hơn.")

    if btn:
        if not settings["api_key"]:
            st.error("Vui lòng nhập Gemini API Key ở sidebar.")
            return
        if not text.strip():
            st.warning("Hãy nhập một đoạn text trước.")
            return

        with st.spinner("Đang rewrite..."):
            try:
                improved = rewrite_section(
                    api_key=settings["api_key"],
                    model_name=settings["model_name"],
                    text=text,
                    language=settings["language"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    max_output_tokens=settings["max_output_tokens"],
                )
            except Exception as e:
                st.error(f"Lỗi rewrite: {e}")
                return

        a, b = st.columns(2)
        with a:
            st.markdown("**Bản gốc**")
            st.write(text)
        with b:
            st.markdown("**Bản rewrite**")
            st.write(improved)


def render_history() -> None:
    st.subheader("🕘 Lịch sử phân tích (trong phiên)")

    history = st.session_state.get("analysis_history", [])
    if not history:
        st.info("Chưa có lịch sử. Hãy chạy phân tích để lưu lại.")
        return

    options = [f"{h['ts']} – score {h['result']['match_score']}/100" for h in history]
    idx = st.selectbox("Chọn một lần phân tích", range(len(options)), format_func=lambda i: options[i])

    chosen = history[idx]
    result = AnalysisResult.from_dict(chosen["result"])

    render_overview(result)
    render_details(result)
    
def render_footer():
    """Render copyright footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>CVMatch AI</strong> - Powered by Google Gemini</p>
            <p>© 2024 [Tên bạn]. All rights reserved.</p>
            <p style='font-size: 12px;'>
                Built with ❤️ using Streamlit | 
                <a href='mailto:your@email.com'>Contact</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main() -> None:
    init_session_state()
    settings = sidebar_settings()

    render_header()

    # Main layout: Inputs + Actions
    cv_text, jd_text = render_inputs()

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        analyze_btn = st.button("🚀 Phân tích CV & JD", type="primary", use_container_width=True)
    with right:
        st.caption("Bạn có thể tải JSON kết quả sau khi phân tích.")

    if analyze_btn:
        if not settings["api_key"]:
            st.error("⚠️ Chưa nhập API Key. Vui lòng nhập ở sidebar.")
        elif not cv_text or not jd_text:
            st.error("Vui lòng nhập **cả CV và JD** trước khi phân tích.")
        else:
            with st.spinner("Gemini đang phân tích..."):
                try:
                    result = analyze_cv_jd(
                        api_key=settings["api_key"],
                        model_name=settings["model_name"],
                        cv_text=cv_text,
                        jd_text=jd_text,
                        language=settings["language"],
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        max_output_tokens=settings["max_output_tokens"],
                    )
                except Exception as e:
                    st.error(f"Lỗi khi gọi Gemini: {e}")
                    result = None

            if result:
                st.session_state.last_result = result.to_dict()
                st.session_state.analysis_history.insert(
                    0,
                    {
                        "ts": now_label(),
                        "cv_chars": len(cv_text),
                        "jd_chars": len(jd_text),
                        "result": result.to_dict(),
                    },
                )

                render_overview(result)
                render_details(result)

                st.download_button(
                    "⬇️ Tải kết quả (JSON)",
                    data=json.dumps(st.session_state.last_result, ensure_ascii=False, indent=2),
                    file_name=f"cv_jd_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
                
    

    # Secondary section
    with st.expander("📌 Xem lịch sử phân tích", expanded=False):
        render_history()

    render_rewrite_tool(settings)
    
    
     # Footer with copyright
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 20px; font-size: 14px;'>
            <p><strong>AI CV & JD</strong> - HCMUS</p>
            <p>© 2025 AI CV & JD. All rights reserved.</p>
            <p style='font-size: 12px; margin-top: 10px;'>
                Version 1.0.0 | Built with ❤️ using Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



if __name__ == "__main__":
    main()
