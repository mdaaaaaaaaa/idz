import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import dns.resolver
import whois
from datetime import datetime

# Загрузка данных
data = pd.read_csv('urldata.csv')

# Преобразование данных
X = data.drop(columns=['Domain', 'Label'])
y = data['Label']  # Метки (0 - безопасный, 1 - фишинговый)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели XGBoost
model = XGBClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print("\nОценка модели:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def check_full_url(url):
    """Проверяет, является ли URL полным (http/https)."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        raise ValueError("Введите полный URL (начиная с 'http://' или 'https://').")

def check_site_exists(domain):
    """Проверяет, существует ли сайт по его домену."""
    try:
        dns.resolver.resolve(domain, 'A')
        return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.Timeout):
        return False

def extract_features(url):
    try:
        url = check_full_url(url)  # Проверяем, полный ли URL
        parts = url.split('/')
        domain = parts[2] if len(parts) > 2 else ""

        if not check_site_exists(domain):
            raise ValueError("Указанный сайт не существует.")

        dns_record = check_dns_record(domain)
        domain_age = get_domain_age(domain)

        return {
            "Have_IP": int(any(char.isdigit() for char in domain)),
            "Have_At": int("@" in url),
            "URL_Length": len(url),
            "URL_Depth": len(parts) - 1,
            "Redirection": int(url.count('//') > 1),
            "https_Domain": int(parts[0].lower() == "https:"),
            "TinyURL": int(len(domain) < 10),
            "Prefix/Suffix": int("-" in domain),
            "DNS_Record": dns_record,
            "Web_Traffic": 1,
            "Domain_Age": domain_age,  
            "Domain_End": int(domain.endswith(('.com', '.org', '.net'))),
            "iFrame": 0,
            "Mouse_Over": 0,
            "Right_Click": 0,
            "Web_Forwards": 0
        }
    except ValueError as ve:
        raise ve  # Пробрасываем ошибку для обработки в графическом интерфейсе
    except Exception as e:
        print(f"Ошибка при обработке URL: {url}. Ошибка: {e}")
        return {key: 0 for key in ["Have_IP", "Have_At", "URL_Length", "URL_Depth", "Redirection", "https_Domain", 
                                  "TinyURL", "Prefix/Suffix", "DNS_Record", "Web_Traffic", "Domain_Age", 
                                  "Domain_End", "iFrame", "Mouse_Over", "Right_Click", "Web_Forwards"]}

def check_dns_record(domain):
    try:
        dns.resolver.resolve(domain, 'A')
        return 1  # DNS-запись существует
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
        return 0  # DNS-запись не существует

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        registration_date = w.creation_date
        if isinstance(registration_date, list):
            registration_date = registration_date[0]

        if registration_date:
            age = (datetime.now() - registration_date).days / 365
            return int(age)
        else:
            return 0
    except Exception as e:
        print(f"Ошибка при получении возраста домена для {domain}: {e}")
        return 0

def classify_url(url):
    try:
        feature_dict = extract_features(url)
        feature_list = [feature_dict[col] for col in X.columns]
        features = pd.DataFrame([feature_list], columns=X.columns)
        prediction = model.predict(features)[0]

        result = "Phishing" if prediction == 1 else "Safe"

        feature_details = "\n".join([f"{key}: {value}" for key, value in feature_dict.items()])

        return result, feature_details
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Произошла ошибка: {e}")

# Создание графического интерфейса
def on_classify():
    url = url_entry.get().strip()
    if not url:
        messagebox.showwarning("Ошибка", "Введите URL для классификации!")
        return

    try:
        result, feature_details = classify_url(url)
        messagebox.showinfo(
            "Результат классификации",
            f"URL: {url}\nРезультат: {result}\n\nПризнаки:\n{feature_details}"
        )
    except ValueError as ve:
        messagebox.showerror("Ошибка", str(ve))
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

# Инициализация интерфейса
root = tk.Tk()
root.title("Классификация URL")

# Поле для ввода URL
url_label = tk.Label(root, text="Введите URL:")
url_label.pack(pady=5)

url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# Кнопка для запуска классификации
classify_button = tk.Button(root, text="Классифицировать", command=on_classify)
classify_button.pack(pady=10)

# Запуск интерфейса
root.mainloop()
