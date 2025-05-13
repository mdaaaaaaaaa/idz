from fastapi import FastAPI
import joblib
import torch
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Добавляем путь к папке с моделями
sys.path.append(str(Path(__file__).parent))

try:
    from train_models import PhishingNN
except ImportError:
    print("Не удалось импортировать PhishingNN. Убедитесь, что train_models.py существует")

def load_model(model_type):
    """Загрузка модели по типу"""
    model_path = f"models/{model_type}_model.{'pkl' if model_type == 'xgboost' else 'pt'}"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель {model_type} не найдена")
    
    if model_type == "xgboost":
        return joblib.load(model_path)
    else:
        # Загружаем checkpoint с размером входа
        checkpoint = torch.load(model_path)
        input_size = checkpoint['input_size']
        
        model = PhishingNN(input_size)
        checkpoint = torch.load("models/pytorch_model.pt", weights_only=False)
        scaler = StandardScaler()
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']

        model = PhishingNN(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

# Загружаем модели при старте
try:
    models = {
        "xgboost": load_model("xgboost"),
        "pytorch": load_model("pytorch")
    }
except Exception as e:
    print(f"Ошибка загрузки моделей: {str(e)}")
    models = {}

class RequestData(BaseModel):
    features: list
    model_type: str  # "xgboost" или "pytorch"

@app.post("/predict")
def predict(data: RequestData):
    if data.model_type == "pytorch":
        features = scaler.transform(np.array(data.features).reshape(1, -1))
        features = torch.FloatTensor(features)
    
    try:
        features = np.array(data.features).reshape(1, -1)
        
        if data.model_type == "xgboost":
            prediction = models["xgboost"].predict(features)[0]
        else:
            # Загружаем scaler
            scaler = joblib.load("models/scaler.pkl")
            features_scaled = scaler.transform(features)
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                prediction = (models["pytorch"](features_tensor).numpy()[0][0] > 0.5).astype(int)
        
        return {"is_phishing": bool(prediction)}
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.get("/models")
def list_models():
    return {"available_models": list(models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)