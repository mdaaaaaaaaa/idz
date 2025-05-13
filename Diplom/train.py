import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from pathlib import Path

# Создаем папку для моделей
Path("models").mkdir(exist_ok=True)

# 1. Загрузка и подготовка данных
def load_data():
    df = pd.read_csv("data/urldata.csv")
    
    # Кодируем категориальные признаки
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop("Label", axis=1)
    y = df["Label"]
    return X, y

# 2. Обучение XGBoost
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    print(f"XGBoost Accuracy: {model.score(X_test, y_test):.4f}")
    joblib.dump(model, "models/xgboost_model.pkl")

# 3. Обучение PyTorch модели
class Net(nn.Module):
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

def train_pytorch(X, y):
    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Конвертация в тензоры
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1)
    
    # Модель и оптимизатор
    model = Net(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), "models/pytorch_model.pt")

if __name__ == "__main__":
    X, y = load_data()
    train_xgboost(X, y)
    train_pytorch(X, y)