import streamlit as st
import requests
import pandas as pd

st.title("🔍 Phishing Website Detector")
st.markdown("""
Проверьте URL на фишинг с помощью ML моделей
""")

# Получаем список доступных моделей
try:
    models = requests.get("http://localhost:8000/models").json()["available_models"]
    model_type = st.selectbox("Выберите модель:", models)
except:
    st.error("Не удалось подключиться к серверу моделей")
    st.stop()

# Форма для ввода признаков
with st.form("prediction_form"):
    st.subheader("Параметры URL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        url_length = st.number_input("Длина URL", min_value=0, value=100)
        url_depth = st.number_input("Глубина URL", min_value=0, value=3)
        has_ip = st.selectbox("Содержит IP", [0, 1], format_func=lambda x: "Да" if x else "Нет")
    
    with col2:
        has_at = st.selectbox("Содержит '@'", [0, 1], format_func=lambda x: "Да" if x else "Нет")
        https = st.selectbox("HTTPS", [0, 1], format_func=lambda x: "Да" if x else "Нет")
        web_traffic = st.slider("Уровень трафика", 0, 100, 50)
    
    submitted = st.form_submit_button("Проверить")

if submitted:
    features = [
        url_length, url_depth, has_ip, has_at, 
        https, web_traffic, 0, 0, 0, 0  # Добавьте остальные признаки
    ]
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features, "model_type": model_type}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["is_phishing"]:
                st.error("⚠️ Фишинговый сайт!")
            else:
                st.success("✅ Безопасный сайт")
        else:
            st.error(f"Ошибка: {response.text}")
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")