import streamlit as st
import requests
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Phishing Detector", layout="wide")

def main():
    st.title("🔍 Phishing Website Detector")
    st.markdown("Проверка URL на фишинг с помощью ML моделей")

    # Проверяем доступность сервера
    try:
        models_response = requests.get("http://localhost:8000/models", timeout=3)
        if models_response.status_code == 200:
            available_models = models_response.json()["available_models"]
        else:
            st.error("Сервер моделей недоступен")
            st.stop()
    except requests.exceptions.RequestException:
        st.error("Не удалось подключиться к серверу. Убедитесь, что api.py запущен")
        st.stop()

    model_type = st.selectbox("Выберите модель:", available_models)

    with st.form("prediction_form"):
        st.subheader("Введите параметры URL")
        
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
            https, web_traffic, 0, 0, 0, 0  # Дополните своими признаками
        ]
        
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={
                    "features": features,
                    "model_type": model_type
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    st.error(f"Ошибка: {result['error']}")
                else:
                    if result["is_phishing"]:
                        st.error("⚠️ Внимание! Возможен фишинг!")
                    else:
                        st.success("✅ Сайт безопасен")
            else:
                st.error(f"Ошибка сервера: {response.text}")
        except Exception as e:
            st.error(f"Ошибка соединения: {str(e)}")

if __name__ == "__main__":
    main()