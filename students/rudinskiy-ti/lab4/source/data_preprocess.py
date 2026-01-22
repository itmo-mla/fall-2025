import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(addr, dataset):
    """
    Загрузка датасета
    
    :param addr: Адрес датасета
    :param dataset: Наименование датасета
    """
    path = kagglehub.dataset_download(addr)
    data = pd.read_csv(path + dataset)
    return data

def encode_skills_by_categories(df, cat, col='primary_skills'):
    """
    Преобразует строковый столбец с навыками в бинарные признаки по заданным категориям.
    
    :param df: Датасет
    :param cat: Словарь с категориями
    :param col: Колонка со списком умений
    """
    df = df.copy()  

    category_sets = {name: set(skills) for name, skills in cat.items()}
    category_names = list(cat.keys())

    def map_row_to_categories(skills_str):
        if not isinstance(skills_str, str):
            return [0] * len(category_names)
        
        skills = {s.strip() for s in skills_str.split(',') if s.strip()}
        
        return [1 if skills & category_sets[name] else 0 for name in category_names]

    binary_rows = df[col].apply(map_row_to_categories).tolist()

    for i, name in enumerate(category_names):
        df[name] = [row[i] for row in binary_rows]

    df.drop(columns=[col], inplace=True)
    return df

def change_income_from_str_to_float(df, col='annual_income_usd'):
    """
    Перевод из строкового формата зарплаты в числовой
    
    :param df: Датасет
    :param col: Колонка с зарплатой в формате $xxx,xxx
    """
    df_copy = df.copy()
    income = df_copy[col].to_numpy()
    for i in range(len(income)):
        income[i] = income[i].replace('$','')
        income[i] = income[i].replace(',','')
        income[i] = float(income[i])
    df_copy['annual_income_usd'] = np.array(income).astype(np.float64)
    return df_copy

def df_label_encoder(df, cols):
    """
    Кодировка категориальных признаков
    
    :param df: Датасет
    :param cols: Список колонок с категориальными переменными
    """
    df_copy = df.copy()
    for i in cols:
        encoder = LabelEncoder()
        df_copy[i] = encoder.fit_transform(df[i])
    return df_copy

def prepare_dataset():
    df = load_data("shaistashahid/freelancer-income-vs-skills", '/freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv')
    categories = {
        "Языки программирования": {
            "Python", "Java", "JavaScript", "Go", "C#", "PHP", "Ruby", "Kotlin", "Swift", "SQL", "Solidity"
        },
        "Фреймворки и библиотеки": {
            "React", "Angular", "Vue.js", "Node.js", "React Native", "Flutter", "TensorFlow", "PyTorch", "GraphQL"
        },
        "СУБД / Хранилища данных": {
            "MongoDB", "PostgreSQL", "BigQuery", "SQL"
        },
        "DevOps / Инфраструктура": {
            "Docker", "Kubernetes", "Terraform", "CI/CD", "AWS", "Azure", "ETL", "Data Warehousing"
        },
        "Data Science / ML / AI": {
            "Data Science", "NLP", "Computer Vision", "PyTorch", "TensorFlow"
        },
        "Безопасность": {
            "Penetration Testing", "Security Audit", "Network Security", "Cryptography"
        },
        "Web3 / Блокчейн": {
            "Web3", "Ethereum", "DeFi", "Smart Contracts", "Solidity"
        },
        "UI/UX / Дизайн / Прототипирование": {
            "Figma", "Sketch", "Adobe XD", "Prototyping", "Wireframing", "User Research"
        },
        "Мобильная разработка": {
            "Android", "iOS", "Kotlin", "Swift", "React Native", "Flutter"
        },
        "Бэкенд / API": {
            "REST API", "GraphQL", "Node.js", "PHP", "Java", "C#"
        },
        "Big Data / Обработка данных": {
            "Spark", "ETL", "BigQuery", "Data Warehousing"
        }
    }
    cols = ['category', 'experience_level', 'region', 'country', 'education', 'primary_platform']
    df = encode_skills_by_categories(df, categories)
    df = change_income_from_str_to_float(df)
    df = df_label_encoder(df, cols)
    df.drop(['freelancer_id', 'hourly_rate_usd'], axis='columns', inplace=True)
    return df
