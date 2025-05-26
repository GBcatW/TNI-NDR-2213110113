import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ------------------------ 🎨 Page Config ------------------------
st.set_page_config(page_title="CPF Price Trend", page_icon="📈", layout="wide")

# ------------------------ 📁 Load Data ------------------------
df = pd.read_excel("CPF.xlsx", sheet_name="CPF", skiprows=1)
df.columns = ["วันที ่", "ราคาเปิด", "ราคาสูงสุด", "ราคาต่ำสุด", "ราคาเฉลี่ย", "ราคาปิด",
              "เปลี่ยนแปลง", "เปลี่ยนแปลง(%)", "ปริมาณ(พันหุ้น)", "มูลค่า(ล้านบาท)",
              "SET Index", "SET เปลี่ยนแปลง(%)"]

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

# ------------------------ 🎛️ Sidebar ------------------------
st.sidebar.title("🛠️ การตั้งค่า")
start_date = st.sidebar.date_input("📅 วันที่เริ่มต้น", df["วันที ่"].min())
end_date = st.sidebar.date_input("📅 วันที่สิ้นสุด", df["วันที ่"].max())
degree = st.sidebar.slider("🎚️ ระดับ Polynomial Regression", 1, 2, 1)
show_table = st.sidebar.checkbox("📋 แสดงตารางข้อมูล", value=False)

# ------------------------ 📊 Filter Data ------------------------
filtered_df = df[(df["วันที ่"] >= pd.to_datetime(start_date)) & (df["วันที ่"] <= pd.to_datetime(end_date))]
X = filtered_df["วันที ่"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = filtered_df["ราคาปิด"].values

# ------------------------ 🤖 Train Model ------------------------
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
trend = model.predict(X_poly)

# ------------------------ 📈 Plot ------------------------
st.title("📈 CPF Closing Price Trend")
st.markdown("### 🔍 วิเคราะห์ราคาปิดหุ้น CPF ด้วย Polynomial Regression")

fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_df["วันที ่"], y=y, mode="lines+markers", name="ราคาจริง", line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=filtered_df["วันที ่"], y=trend, mode="lines", name=f"แนวโน้ม (Deg {degree})",
                         line=dict(color="red", dash="dash")))
fig.update_layout(
    title="CPF Interactive Closing Price Trend",
    xaxis_title="วันที่",
    yaxis_title="ราคาปิด (บาท)",
    template="plotly_white",
    legend_title="คำอธิบาย",
    width=1000, height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------ 📊 สถิติ ------------------------
st.subheader("📌 สถิติเบื้องต้นของราคาปิด")
col1, col2, col3 = st.columns(3)
col1.metric("🎯 ราคาเฉลี่ย", f"{filtered_df['ราคาปิด'].mean():.2f} บาท")
col2.metric("⬆️ สูงสุด", f"{filtered_df['ราคาปิด'].max():.2f} บาท")
col3.metric("⬇️ ต่ำสุด", f"{filtered_df['ราคาปิด'].min():.2f} บาท")

# ------------------------ 📋 แสดงตาราง ------------------------
if show_table:
    st.markdown("### 📋 ตารางข้อมูล")
    st.dataframe(filtered_df, use_container_width=True)

# ------------------------ 💾 Download ------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

csv = convert_df_to_csv(filtered_df)
st.download_button("📥 ดาวน์โหลดข้อมูลเป็น CSV", csv, "cpf_filtered_data.csv", "text/csv")

# ------------------------ 📝 หมายเหตุ ------------------------
st.markdown("---")
st.markdown("💡 **หมายเหตุ:** การวิเคราะห์นี้ใช้โมเดล Polynomial Regression เพื่อหาแนวโน้มราคาหุ้น CPF ในช่วงเวลาที่คุณเลือก")

