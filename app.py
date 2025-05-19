import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the Excel data
df = pd.read_excel("PTT-SET-19May2025-6M.xlsx", sheet_name="PTT", skiprows=1)
df.columns = ["วันที ่", "ราคาเปิด", "ราคาสูงสุด", "ราคาต่ำสุด", "ราคาเฉลี่ย", "ราคาปิด", 
              "เปลี่ยนแปลง", "เปลี่ยนแปลง(%)", "ปริมาณ(พันหุ้น)", "มูลค่า(ล้านบาท)", 
              "SET Index", "SET เปลี่ยนแปลง(%)"]

# Data preprocessing (Thai date handling, sorting, etc.)
thai_months = {
    "ม.ค.": "01", "ก.พ.": "02", "มี.ค.": "03", "เม.ย.": "04",
    "พ.ค.": "05", "มิ.ย.": "06", "ก.ค.": "07", "ส.ค.": "08",
    "ก.ย.": "09", "ต.ค.": "10", "พ.ย.": "11", "ธ.ค.": "12"
}

def convert_thai_date(thai_date_str):
    for th, num in thai_months.items():
        if th in thai_date_str:
            day, month_th, year_th = thai_date_str.replace(",", "").split()
            month = thai_months[month_th]
            year = int(year_th) - 543
            return f"{year}-{month}-{int(day):02d}"
    return None

df["วันที ่"] = df["วันที ่"].apply(convert_thai_date)
df["วันที ่"] = pd.to_datetime(df["วันที ่"], errors='coerce')
df = df.dropna(subset=["วันที ่"])

df = df.sort_values("วันที ่")
X = df["วันที ่"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df["ราคาปิด"].values

# Linear regression model
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# Streamlit components
st.title("PTT Closing Price Trend")
st.write("### Linear Regression Analysis of PTT Closing Prices")

# Make table wider by specifying width
st.dataframe(df, width=2200)

fig, ax = plt.subplots()
ax.plot(df["วันที ่"], y, label="Actual")
ax.plot(df["วันที ่"], trend, label="Trend", color="red", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (Baht)")
ax.legend()
ax.grid()

st.pyplot(fig)