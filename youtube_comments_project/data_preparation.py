import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import re

# Здесь мы бы использовали API Kaggle для скачивания датасета
# Но, поскольку требуется авторизация, мы предполагаем, что датасет уже скачан

def clean_text(text):
    """Очистка текста комментариев"""
    if isinstance(text, str):
        # Удаление HTML-тегов
        text = re.sub(r'<.*?>', '', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление URL
        text = re.sub(r'http\S+', '', text)
        return text.strip()
    return ''

def prepare_data(data_path, output_path, max_length=128):
    """Подготовка данных для обучения модели"""
    print(f"Загрузка данных из {data_path}...")
    
    # Загрузка данных
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Неподдерживаемый формат файла. Используйте CSV или JSON.")
    
    # Предположим, что у нас есть колонка 'text' с комментариями
    if 'text' not in df.columns and 'comment_text' in df.columns:
        df['text'] = df['comment_text']
    elif 'text' not in df.columns and 'content' in df.columns:
        df['text'] = df['content']
    
    if 'text' not in df.columns:
        print("Колонка с текстом комментариев не найдена.")
        print("Доступные колонки:", df.columns.tolist())
        return
    
    # Очистка текста
    print("Очистка текста комментариев...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Удаление пустых комментариев
    df = df[df['clean_text'].str.len() > 5]
    
    # Разделение на тренировочную и тестовую выборки
    train_size = int(len(df) * 0.9)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Сохранение данных
    os.makedirs(output_path, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    
    print(f"Обработано {len(df)} комментариев.")
    print(f"Сохранено {len(train_df)} тренировочных и {len(test_df)} тестовых примеров.")
    print(f"Данные сохранены в {output_path}")

if __name__ == "__main__":
    # Здесь нужно указать путь к скачанному датасету
    # Предположим, что он находится в папке data/raw
    data_dir = "data/raw"
    output_dir = "data/processed"
    
    # Имя файла датасета может отличаться
    dataset_path = os.path.join(data_dir, "youtube_comments.csv")
    
    # Создаем директории, если их нет
    os.makedirs(data_dir, exist_ok=True)
    
    prepare_data(dataset_path, output_dir) 