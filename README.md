# AI Analyzes CV & Job Descriptions 📄

Ứng dụng phân tích độ phù hợp giữa CV và Job Description 

## ✨ Tính năng

- 🎯 **Phân tích độ phù hợp**: Tính điểm match score (0-100) giữa CV và JD
- 📊 **Xác định Seniority**: Tự động nhận diện cấp độ kinh nghiệm (Intern/Junior/Mid/Senior/Lead)
- 💪 **Phát hiện điểm mạnh**: Liệt kê các kỹ năng và kinh nghiệm nổi bật
- ⚠️ **Tìm khoảng trống kỹ năng**: Chỉ ra những gì còn thiếu so với JD
- 🔑 **Gợi ý từ khóa**: Đề xuất keywords cần bổ sung vào CV
- ✏️ **Rewrite thông minh**: Cải thiện các bullet points trong CV
- 📁 **Hỗ trợ nhiều định dạng**: Upload file PDF, DOCX, TXT hoặc dán text trực tiếp
- 🕘 **Lịch sử phân tích**: Lưu và xem lại các lần phân tích trước
- 💾 **Export JSON**: Tải xuống kết quả phân tích

## 📁 Cấu trúc project

```
AI CV & JD/
│
├── .streamlit/
│   └── config.toml              # Cấu hình Streamlit
│
├── AI Application - dashboard.pdf   # Bản phân công task
├── AI CV & JD.xlsx                # Kế hoạch thực hiện
├── app.py                           # source app
└── System Architecture Diagram.png  # Sơ đồ kiến trúc hệ thống
```

## 🚀 Hướng dẫn chạy nhanh (Quick Start)

### Bước 1: Cài đặt Python

**Yêu cầu**: Python 3.8 trở lên

**Kiểm tra Python:**
```bash
python --version
```

**Nếu chưa có Python:**
- Windows: Tải từ [python.org](https://www.python.org/downloads/)
- macOS: `brew install python@3.11`
- Linux: `sudo apt install python3.11`

### Bước 2: Cài đặt thư viện

Mở Terminal/Command Prompt tại thư mục `AI CV & JD`, chạy:

```bash
pip install streamlit google-generativeai PyPDF2 python-docx
```

Hoặc nếu có file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở tại: **http://localhost:8501**

## 📖 Hướng dẫn sử dụng chi tiết

### 1. Nhập CV của bạn

Có 2 cách:

**Cách 1: Upload file** 
- Click "Upload file"
- Chọn file CV: `.pdf`, `.docx`, hoặc `.txt`
- Hệ thống tự động đọc nội dung

**Cách 2: Dán text**
- Click "Dán text"
- Copy toàn bộ nội dung CV
- Paste vào ô text

### 2. Nhập Job Description (JD)

**Cách 1: Upload file JD**
- Click "Upload file"
- Chọn file JD (PDF/DOCX/TXT)

**Cách 2: Dán text JD**
- Copy JD từ website tuyển dụng
- Paste vào ô text

### 3. Phân tích

- Click nút **"🚀 Phân tích CV & JD"**
- Đợi 5-15 giây AI xử lý
- Xem kết quả chi tiết

### 4. Đọc kết quả

#### 📊 Tổng quan
- **Match Score**: Điểm phù hợp (0-100)
- **Seniority**: Cấp độ kinh nghiệm
- **Tóm tắt CV & JD**

#### Chi tiết 4 tab

**✅ Tab Điểm mạnh**
- Những gì CV đã làm tốt
- Kỹ năng nổi bật
- Kinh nghiệm liên quan

**⚠️ Tab Khoảng trống**
- Kỹ năng còn thiếu
- Kinh nghiệm chưa đủ
- Yêu cầu JD chưa đáp ứng

**🧩 Tab Từ khóa nên thêm**
- Keywords để tối ưu ATS
- Thuật ngữ chuyên ngành
- Skills cần highlight

**✏️ Tab Ví dụ rewrite**
- So sánh bullet gốc vs cải thiện
- Gợi ý viết lại chuyên nghiệp
- Action verbs và số liệu cụ thể

### 5. Rewrite nhanh 1 đoạn CV

- Scroll xuống phần **"✨ Rewrite nhanh một đoạn CV"**
- Dán đoạn text cần cải thiện
- Click **"Rewrite ✏️"**
- So sánh bản gốc và bản được AI tối ưu

### 6. Lưu kết quả

- Click **"⬇️ Tải kết quả (JSON)"**
- File JSON chứa toàn bộ phân tích
- Dùng để tham khảo hoặc so sánh sau

## ⚙️ Cấu hình (Tùy chọn)

### API Key đã được cài sẵn
Bạn không cần làm gì thêm, app sử dụng API key có sẵn.

### Sidebar Settings

**Ngôn ngữ output:**
- Tiếng Việt (mặc định)
- English

**Tùy chỉnh nâng cao** (Click vào expander):
- **Temperature** (0.0 - 1.0): Độ sáng tạo của AI
  - 0.0 = Chính xác, cứng nhắc
  - 1.0 = Sáng tạo, đa dạng
  - Mặc định: 0.7
  
- **Top-p** (0.1 - 1.0): Độ đa dạng từ ngữ
  - Mặc định: 0.95
  
- **Max output tokens** (512 - 8192): Độ dài response
  - Mặc định: 4096

## 🔧 Xử lý lỗi thường gặp

### ❌ Lỗi: "ModuleNotFoundError: No module named 'streamlit'"

**Nguyên nhân**: Chưa cài thư viện

**Giải pháp**:
```bash
pip install streamlit google-generativeai PyPDF2 python-docx
```

### ❌ Lỗi: "python is not recognized"

**Nguyên nhân**: Python chưa được thêm vào PATH

**Giải pháp**:
- Cài lại Python, nhớ tick "Add Python to PATH"
- Hoặc dùng: `py` thay vì `python`

### ❌ Lỗi: "Cannot read PDF/DOCX file"

**Nguyên nhân**: File bị mã hóa hoặc hỏng

**Giải pháp**:
- Kiểm tra file không bị password
- Thử convert sang PDF khác hoặc dùng "Dán text"
- Đảm bảo file không quá 50MB

### ❌ Lỗi: "Gemini API error" / "Rate limit exceeded"

**Nguyên nhân**: 
- Mất kết nối internet
- API key hết quota
- Request quá nhiều

**Giải pháp**:
- Kiểm tra internet
- Đợi vài phút rồi thử lại
- Giảm độ dài CV/JD (tối đa ~120,000 ký tự)

### ❌ App không mở trên browser

**Giải pháp**:
- Thử mở thủ công: http://localhost:8501
- Hoặc: http://127.0.0.1:8501
- Kiểm tra port 8501 có bị chiếm không

## 💡 Tips sử dụng hiệu quả

### Để CV:
1. ✅ Có cấu trúc rõ ràng (Experience, Skills, Education)
2. ✅ Dùng bullet points thay đoạn văn dài
3. ✅ Thêm số liệu cụ thể: "Tăng doanh thu 30%" > "Tăng doanh thu"
4. ✅ Dùng action verbs: Led, Developed, Managed, Optimized
5. ✅ Đầy đủ 1-2 trang A4

### Để JD:
1. ✅ Copy toàn bộ JD từ website tuyển dụng
2. ✅ Bao gồm: Requirements, Responsibilities, Nice-to-have
3. ✅ Không cắt bớt thông tin quan trọng

### Đọc kết quả:
- **Match Score >= 70**: ✅ Tốt, nên apply
- **Match Score 50-69**: ⚠️ Khá, cần improve CV
- **Match Score < 50**: ❌ Chưa phù hợp, cần bổ sung nhiều

## 📚 Tài liệu tham khảo

- **AI Application - dashboard.pdf**: Hướng dẫn chi tiết về app
- **System Architecture Diagram.png**: Sơ đồ kiến trúc hệ thống
- **AI CV & JD.xlsx**: Dữ liệu  

## 🌐 Deploy lên Streamlit Cloud (Tùy chọn)

Nếu muốn chia sẻ app online:

1. Push code lên GitHub
2. Truy cập [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect với GitHub repository
4. Click Deploy (miễn phí)
5. App sẽ có URL riêng để chia sẻ

## 🛠️ Development

### Chạy ở chế độ auto-reload
```bash
streamlit run app.py --server.runOnSave true
```

### Debug mode
```bash
streamlit run app.py --logger.level=debug
```

### Xóa cache
```bash
streamlit cache clear
```
