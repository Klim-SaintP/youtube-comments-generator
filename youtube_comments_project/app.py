import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from pytube import YouTube
import matplotlib.pyplot as plt
from train_model import CommentGenerationModel

# Кэширование загрузки модели
@st.cache_resource
def load_model(model_path="./model_output"):
    """Загрузка обученной модели"""
    model = CommentGenerationModel(model_name=model_path, output_dir=model_path)
    model.load_tokenizer_and_model()
    return model

def extract_video_info(url):
    """Извлечение информации о видео из URL"""
    try:
        # Создание объекта YouTube
        yt = YouTube(url)
        
        # Получение информации о видео
        title = yt.title
        author = yt.author
        description = yt.description
        thumbnail_url = yt.thumbnail_url
        views = yt.views
        length = yt.length
        
        # Проверка языка (возможно потребуется дополнительная логика)
        # Для простоты предполагаем, что видео на английском
        is_english = True
        
        return {
            "title": title,
            "author": author,
            "description": description,
            "thumbnail_url": thumbnail_url,
            "views": views,
            "length": length,
            "is_english": is_english
        }
    except Exception as e:
        st.error(f"Ошибка при обработке URL: {str(e)}")
        return None

def generate_usernames(num_users=5):
    """Генерация случайных имен пользователей"""
    first_names = ["John", "Emma", "Michael", "Sophia", "David", "Olivia", 
                  "James", "Ava", "Robert", "Isabella", "Alex", "Maria", 
                  "Sam", "Jessica", "Daniel", "Lisa", "Max", "Sarah"]
    
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", 
                 "Davis", "Wilson", "Anderson", "Taylor", "Thomas", "Jackson", 
                 "White", "Harris", "Martin", "Lewis", "Clark", "Lee"]
    
    usernames = []
    for _ in range(num_users):
        if np.random.random() < 0.5:
            # Имя + Фамилия
            username = np.random.choice(first_names) + np.random.choice(last_names)
        else:
            # Имя + числа
            username = np.random.choice(first_names) + str(np.random.randint(10, 1000))
        
        usernames.append(username)
    
    return usernames

def format_comments(comments, generate_replies=True):
    """Форматирование сгенерированных комментариев"""
    formatted_comments = []
    usernames = generate_usernames(len(comments))
    
    for i, (comment, username) in enumerate(zip(comments, usernames)):
        comment_data = {
            "username": username,
            "comment": comment.strip(),
            "likes": np.random.randint(0, 1000),
            "time": f"{np.random.randint(1, 30)} дней назад",
            "replies": []
        }
        
        # Генерация ответов на комментарии (для некоторых комментариев)
        if generate_replies and np.random.random() < 0.6:
            num_replies = np.random.randint(1, 3)
            reply_usernames = generate_usernames(num_replies)
            
            for j in range(num_replies):
                reply = {
                    "username": reply_usernames[j],
                    "comment": f"Отвечая на комментарий @{username}: Совершенно согласен! {comment.split(' ')[:5]}..." if np.random.random() < 0.5 else f"Интересная точка зрения, но я думаю иначе.",
                    "likes": np.random.randint(0, 100),
                    "time": f"{np.random.randint(1, 20)} дней назад"
                }
                comment_data["replies"].append(reply)
        
        formatted_comments.append(comment_data)
    
    return formatted_comments

def main():
    # Настройка страницы
    st.set_page_config(
        page_title="YouTube Comment Generator",
        page_icon="🎬",
        layout="wide"
    )
    
    # Заголовок
    st.title("🎬 Генератор комментариев для YouTube")
    st.markdown("Введите ссылку на англоязычное видео YouTube и получите сгенерированные комментарии")
    
    # Боковая панель с инструкциями
    with st.sidebar:
        st.header("О проекте")
        st.markdown("""
        Этот инструмент использует модель GPT-2 large, обученную на комментариях YouTube,
        для генерации реалистичных комментариев к видео.
        
        **Как использовать:**
        1. Введите URL-адрес видео YouTube
        2. Настройте параметры генерации
        3. Нажмите 'Сгенерировать комментарии'
        
        **Настройки генерации:**
        - **Количество комментариев**: Сколько комментариев сгенерировать
        - **Temperature**: Контролирует креативность (выше = более случайно)
        - **Top-p**: Вероятностный порог для генерации (ниже = более консервативно)
        - **Максимальная длина**: Максимальная длина комментария
        """)
    
    # Ввод URL видео
    url = st.text_input("Введите URL видео YouTube:", help="Например: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    # Параметры генерации
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_comments = st.slider("Количество комментариев:", min_value=1, max_value=20, value=5)
    
    with col2:
        temperature = st.slider("Temperature:", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    
    with col3:
        top_p = st.slider("Top-p:", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
    
    with col4:
        max_length = st.slider("Максимальная длина:", min_value=50, max_value=300, value=100, step=10)
        generate_replies_enabled = st.checkbox("Генерировать ответы", value=True)
    
    # Сгенерировать комментарии
    if st.button("Сгенерировать комментарии"):
        if url:
            with st.spinner("Извлечение информации о видео..."):
                video_info = extract_video_info(url)
            
            if video_info and video_info["is_english"]:
                # Отображение информации о видео
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(video_info["thumbnail_url"], caption="Превью видео")
                
                with col2:
                    st.subheader(video_info["title"])
                    st.markdown(f"**Автор:** {video_info['author']}")
                    st.markdown(f"**Просмотры:** {video_info['views']:,}")
                    st.markdown(f"**Длительность:** {video_info['length'] // 60} мин {video_info['length'] % 60} сек")
                
                # Укороченное описание видео
                short_description = video_info["description"][:300] + "..." if len(video_info["description"]) > 300 else video_info["description"]
                with st.expander("Описание видео"):
                    st.markdown(short_description)
                
                # Загрузка модели
                try:
                    with st.spinner("Загрузка модели..."):
                        model = load_model()
                    
                    # Подготовка промпта на основе видео
                    prompt = f"<BOS>This video about {video_info['title']}"
                    
                    # Генерация комментариев
                    with st.spinner("Генерация комментариев..."):
                        # Генерация комментариев с визуализацией процесса
                        placeholder = st.empty()
                        comments = []
                        
                        for i in range(num_comments):
                            # Генерация одного комментария
                            comment = model.generate_comments(
                                prompt=prompt,
                                max_length=max_length,
                                num_return_sequences=1,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True
                            )[0]
                            
                            comments.append(comment)
                            
                            # Обновление UI в реальном времени
                            placeholder.markdown(f"Сгенерировано {i+1}/{num_comments} комментариев...")
                        
                        # Форматирование комментариев
                        formatted_comments = format_comments(comments, generate_replies=generate_replies_enabled)
                        
                        # Очистка плейсхолдера
                        placeholder.empty()
                    
                    # Вывод комментариев
                    st.subheader("Сгенерированные комментарии:")
                    
                    for i, comment_data in enumerate(formatted_comments):
                        with st.container():
                            st.markdown(f"**{comment_data['username']}** • {comment_data['time']} • 👍 {comment_data['likes']}")
                            st.markdown(comment_data['comment'])
                            
                            # Отображение ответов
                            if comment_data["replies"]:
                                with st.container():
                                    for reply in comment_data["replies"]:
                                        st.markdown(f"↪️ **{reply['username']}** • {reply['time']} • 👍 {reply['likes']}")
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{reply['comment']}")
                            
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"Ошибка при генерации комментариев: {str(e)}")
            
            elif video_info and not video_info["is_english"]:
                st.warning("Это видео, возможно, не на английском языке. Модель обучена на англоязычных комментариях.")
            else:
                st.error("Не удалось извлечь информацию о видео. Пожалуйста, проверьте URL.")
        else:
            st.warning("Пожалуйста, введите URL видео YouTube.")

if __name__ == "__main__":
    main() 