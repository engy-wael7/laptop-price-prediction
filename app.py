import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.title("💻 Laptop Price Prediction App")
st.write("Enter the laptop specifications below to get the predicted price:")

# تحميل الداتا الأصلية للتدريب السريع داخل الـ app
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('laptops.csv')
    df = df.drop_duplicates()
    df = df.dropna(subset=['Final Price'])
    df = df[df['Final Price'] > 0]
    df = df.dropna(subset=['Screen'])
    
    df["has_gpu"] = (df["GPU"].notnull() & (df["GPU"] != "No GPU")).astype(int)
    df["GPU"] = df["GPU"].fillna("No GPU")
    
    X = df.drop(columns=['Final Price'])
    y = df['Final Price']
    
    NUM = ['RAM', 'Storage', 'Screen', 'has_gpu']
    CAT = ['Laptop', 'Status', 'Brand', 'Model', 'CPU', 'Storage type', 'GPU']
    BIN = ['Touch']
    all_cat_cols = CAT + BIN
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUM),
        
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), all_cat_cols)
    ], remainder="drop")
    
    voting_regressor = VotingRegressor(
        estimators=[
            ('ridge', Ridge(alpha=1)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('lr', LinearRegression())
        ]
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", voting_regressor)
    ])
    
    pipeline.fit(X, y)
    return pipeline

with st.spinner("Loading and preparing model... Please wait."):
    model = load_and_train_model()

# واجهة المدخلات للمستخدم
laptop = st.selectbox("Laptop Type / Series", ['General', 'Gaming', 'Ultrabook', 'Workstation'])
status = st.selectbox("Status", ['New', 'Refurbished', 'Used'])
brand = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI'])
model_name = st.text_input("Model", "Inspiron")
cpu = st.selectbox("CPU", ['Intel Core i5', 'Intel Core i7', 'Intel Core i3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'Apple M1'])
ram = st.number_input("RAM (GB)", min_value=2, max_value=128, value=8)
storage = st.number_input("Storage (GB)", min_value=128, max_value=4096, value=512)
storage_type = st.selectbox("Storage Type", ['SSD', 'HDD'])
gpu = st.selectbox("GPU", ['Intel Integrated', 'NVIDIA GTX 1650', 'NVIDIA RTX 3060', 'AMD Radeon'])
screen = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6)
touch = st.selectbox("Touch Screen", ['Yes', 'No'])
has_gpu = st.selectbox("Has Dedicated GPU", [1, 0])

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'Laptop': [laptop],
        'Status': [status],
        'Brand': [brand],
        'Model': [model_name],
        'CPU': [cpu],
        'RAM': [ram],
        'Storage': [storage],
        'Storage type': [storage_type],
        'GPU': [gpu],
        'Screen': [screen],
        'Touch': [touch],
        'has_gpu': [has_gpu]
    })
    
    prediction = model.predict(input_data)
    st.success(f"The predicted price is: ${prediction[0]:,.2f}")
    