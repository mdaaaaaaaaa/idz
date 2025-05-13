import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb  # Добавьте этот импорт
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# --- 1. Класс модели ---
class PhishingNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# --- 2. Функции обучения ---
def load_data():
    df = pd.read_csv("data/urldata.csv")
    
    # Кодирование категориальных признаков
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df.drop("Label", axis=1), df["Label"]

def train_xgboost(X, y):
    model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
    model.fit(X, y)
    joblib.dump(model, "models/xgboost_model.pkl")
    return model

def train_pytorch(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = PhishingNN(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    torch.save({
        'state_dict': model.state_dict(),
        'input_size': X.shape[1],
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, "models/pytorch_model.pt")
    
    return model, scaler

# --- 3. Интерфейс Streamlit ---
def main():
    st.title("🔍 Детектор фишинга")
    
    # Выбор действия
    action = st.sidebar.radio("Выберите действие:", ["Обучение моделей", "Проверка URL"])
    
    if action == "Обучение моделей":
        st.header("Обучение моделей")
        if st.button("Обучить модели"):
            with st.spinner("Обучение..."):
                X, y = load_data()
                
                # XGBoost
                st.write("Обучаем XGBoost...")
                xgb_model = train_xgboost(X, y)
                
                # PyTorch
                st.write("Обучаем нейросеть...")
                pt_model, scaler = train_pytorch(X, y)
                
            st.success("Модели успешно обучены!")
            
    else:  # Проверка URL
        st.header("Проверка URL")
        model_type = st.selectbox("Выберите модель:", ["XGBoost", "PyTorch"])
        
        # Поля для ввода признаков
        url_length = st.number_input("Длина URL", min_value=0, value=100)
        url_depth = st.number_input("Глубина URL", min_value=0, value=3)
        has_ip = st.selectbox("Содержит IP", [0, 1])
        has_at = st.selectbox("Содержит '@'", [0, 1])
        
        if st.button("Проверить"):
            features = [url_length, url_depth, has_ip, has_at]
            
            if model_type == "XGBoost":
                try:
                    model = joblib.load("models/xgboost_model.pkl")
                    pred = model.predict([features])[0]
                except:
                    st.error("Модель XGBoost не найдена!")
                    return
            else:
                try:
                    checkpoint = torch.load("models/pytorch_model.pt", weights_only=False)
                    
                    # Восстанавливаем scaler
                    scaler = StandardScaler()
                    scaler.mean_ = checkpoint['scaler_mean']
                    scaler.scale_ = checkpoint['scaler_scale']
                    
                    # Загружаем модель
                    model = PhishingNN(input_size=checkpoint['input_size'])
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    # Предсказание
                    features_scaled = scaler.transform([features])
                    with torch.no_grad():
                        pred = (model(torch.FloatTensor(features_scaled)) > 0.5).int().item()
                except:
                    st.error("Модель PyTorch не найдена!")
                    return
            
            st.success("✅ Безопасный" if not pred else "⚠️ Фишинг!")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()