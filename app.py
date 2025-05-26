import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ------------------------ 🎨 Page Config ------------------------
st.set_page_config(page_title="CPF Price Trend", page_icon="📈", layout="wide")

# ------------------------ 📁 Load Data ------------------------
df = pd.read_excel("CPF.xlsx", sheet_name="CPF", skiprows=1)

# ทำความสะอาดชื่อคอลัมน์
df.columns = df.columns.str.strip().str.replace(r"\s+", "", regex=True)
df.columns = ["วันที่", "ราคาเปิด", "ราคาสูงสุด", "ราคาต่ำสุด", "ราคาเฉลี่ย", "ราคาปิด",
              "เปลี่ยนแปลง", "เปลี่ยนแปลง(%)", "ปริมาณ(พันหุ้น)", "มูลค่า(ล้านบาท)",
              "SETIndex", "SETเปลี่ยนแปลง(%)"]

# แปลงวันที่ไทย
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

df["วันที่"] = df["วันที่"].apply(convert_thai_date)
df["วันที่"] = pd.to_datetime(df["วันที่"], errors='coerce')
df = df.dropna(subset=["วันที่"])
df = df.sort_values("วันที่")

# ------------------------ 🎛️ Sidebar ------------------------
st.sidebar.title("🛠️ การตั้งค่า")

column_options = ["ราคาปิด", "ราคาเปิด", "ราคาสูงสุด", "ราคาต่ำสุด", "ราคาเฉลี่ย",
                  "ปริมาณ(พันหุ้น)", "มูลค่า(ล้านบาท)"]
selected_column = st.sidebar.selectbox("📊 เลือกประเภทข้อมูลที่ต้องการดูกราฟ", column_options)

start_date = st.sidebar.date_input("📅 วันที่เริ่มต้น", df["วันที่"].min())
end_date = st.sidebar.date_input("📅 วันที่สิ้นสุด", df["วันที่"].max())
degree = st.sidebar.slider("🎚️ ระดับ Polynomial Regression", 1, 2, 1)
show_table = st.sidebar.checkbox("📋 แสดงตารางข้อมูล", value=False)

# ------------------------ 📊 Filter Data ------------------------
filtered_df = df[(df["วันที่"] >= pd.to_datetime(start_date)) & (df["วันที่"] <= pd.to_datetime(end_date))]
X = filtered_df["วันที่"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = filtered_df[selected_column].values

# ------------------------ 🤖 Train Model ------------------------
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
trend = model.predict(X_poly)

# ------------------------ 📈 Plot ------------------------
st.title("📈 CPF Data Trend Viewer")
st.markdown(f"### 🔍 วิเคราะห์ข้อมูล **{selected_column}** ด้วย Polynomial Regression")

fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_df["วันที่"], y=y, mode="lines+markers",
                         name=selected_column, line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=filtered_df["วันที่"], y=trend, mode="lines",
                         name=f"แนวโน้ม (Deg {degree})", line=dict(color="red", dash="dash")))

fig.update_layout(
    title=f"{selected_column} Trend Over Time",
    xaxis_title="วันที่",
    yaxis_title=selected_column,
    template="plotly_white",
    legend_title="คำอธิบาย",
    width=1000, height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------ 📊 สถิติ ------------------------
st.subheader(f"📌 สถิติเบื้องต้นของ {selected_column}")
col1, col2, col3 = st.columns(3)
col1.metric("🎯 ค่าเฉลี่ย", f"{filtered_df[selected_column].mean():,.2f}")
col2.metric("⬆️ สูงสุด", f"{filtered_df[selected_column].max():,.2f}")
col3.metric("⬇️ ต่ำสุด", f"{filtered_df[selected_column].min():,.2f}")

# ------------------------ 📋 แสดงตาราง ------------------------
if show_table:
    st.markdown("### 📋 ตารางข้อมูล")
    st.dataframe(filtered_df[["วันที่", selected_column]], use_container_width=True)

# ------------------------ 💾 Download ------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

csv = convert_df_to_csv(filtered_df[["วันที่", selected_column]])
st.download_button("📥 ดาวน์โหลดข้อมูลเป็น CSV", csv, f"cpf_{selected_column}.csv", "text/csv")

# ------------------------ 📝 หมายเหตุ ------------------------
st.markdown("---")
st.markdown("💡 **หมายเหตุ:** การวิเคราะห์นี้ใช้โมเดล Polynomial Regression เพื่อหาแนวโน้มของข้อมูลที่คุณเลือกในช่วงเวลาที่กำหนด")
