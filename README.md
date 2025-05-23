# 📈 CPF Stock Price Trend Analysis with Linear Regression

โครงการนี้เป็นการวิเคราะห์แนวโน้มราคาปิดของหุ้น CPF โดยใช้โมเดล **Linear Regression** และสร้าง Web Interface ด้วย **Streamlit** เพื่อให้ผู้ใช้สามารถดูกราฟแนวโน้มย้อนหลังได้อย่างสะดวก

## 🔧 เทคโนโลยีที่ใช้
- Python 3
- pandas
- scikit-learn (LinearRegression)
- matplotlib
- Streamlit

## 📊 ข้อมูลที่ใช้
ไฟล์ `CPF.xlsx` ที่มีข้อมูลราคาหุ้นย้อนหลัง เช่น
- ราคาปิด
- ปริมาณการซื้อขาย
- วันที่ (รูปแบบไทย)

## 📌 ฟีเจอร์ที่มีในเว็บ
- แสดงตารางข้อมูลราคาหุ้น CPF
- แปลงวันที่แบบไทย → สากล
- สร้างกราฟราคาปิดพร้อมเส้นแนวโน้ม (Linear Regression)
- ใช้งานได้ผ่านหน้าเว็บ Streamlit

## ▶️ การรันโปรเจกต์

1. ติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install streamlit pandas scikit-learn matplotlib openpyxl
