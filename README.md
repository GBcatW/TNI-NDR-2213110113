# 📈 CPF Stock Price Trend Analysis with Linear Regression

โครงการนี้เป็นการวิเคราะห์แนวโน้มราคาปิดของหุ้น CPF โดยใช้โมเดล **Linear Regression** และสร้าง Web Interface ด้วย **Streamlit** เพื่อให้ผู้ใช้สามารถดูกราฟแนวโน้มย้อนหลังได้อย่างสะดวก

## 🔧 เทคโนโลยีที่ใช้
- Python 3
- pandas
- scikit-learn (LinearRegression)
- matplotlib
- Streamlit
- plotly

## 📊 ข้อมูลที่ใช้
ไฟล์ `CPF.xlsx` ที่มีข้อมูลราคาหุ้นย้อนหลัง เช่น
- ราคาปิด
- ราคาเปิด
- ราคาสูงสุด
- ราคาต่ำสุด
- ราคาเฉลี่ย
- ปริมาณการซื้อขาย
- วันที่ (รูปแบบวันที่ไทย พ.ศ.)

## 📌 ฟีเจอร์ที่มีในเว็บ
- 📅 เลือกช่วงวันที่เริ่มต้น–สิ้นสุดที่ต้องการวิเคราะห์
- 📊 เลือกประเภทข้อมูลที่จะวิเคราะห์ เช่น ราคาปิด, ราคาเปิด, ปริมาณ, ฯลฯ
- 📈 สร้างกราฟแนวโน้มข้อมูลที่เลือก พร้อมเส้น Polynomial Regression (เลือก degree ได้ 1 หรือ 2)
- 🧮 แสดงค่าสถิติเบื้องต้นของข้อมูลที่เลือก (ค่าเฉลี่ย, ค่าสูงสุด, ค่าต่ำสุด)
- 📋 แสดงตารางข้อมูลที่ใช้ในกราฟ
- 💾 ดาวน์โหลดข้อมูลที่ใช้วิเคราะห์เป็นไฟล์ CSV

## 📊 Presentation Slides

ดูสไลด์นำเสนอ: [presentation.pdf](slides/TNI-NDR-2213110113.pdf)

## ▶️ การรันโปรเจกต์

1. ติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install streamlit pandas scikit-learn matplotlib openpyxl

2. รันแอป Streamlit:
   ```bash
   streamlit run app.py
