import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from pytube import YouTube
import matplotlib.pyplot as plt
from train_model import CommentGenerationModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from collections import Counter

# Проверяем, существует ли модель, если нет - запускаем обучение
if not os.path.exists("./model_output/pytorch_model.bin"):
    import train_on_startup
    train_on_startup.train_if_needed()

# Кэширование загрузки модели
@st.cache_resource
def load_model(model_path="./model_output"):
    """Загрузка обученной модели"""
    model = CommentGenerationModel(model_name=model_path, output_dir=model_path)
    model.load_tokenizer_and_model()
    
    # Проверяем, что модель не находится в процессе обучения
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        # Заранее подготавливаем оптимизированную версию модели для инференса
        try:
            model.inference_model = model.optimize_model_for_inference()
        except Exception as e:
            st.warning(f"Не удалось оптимизировать модель: {str(e)}")
    
    return model

@st.cache_data(ttl=3600)
def extract_video_info_cached(url):
    """Кэшированное извлечение информации о видео"""
    return extract_video_info(url)

@st.cache_data(ttl=3600)
def generate_comments_cached(_model, prompt, generation_params):
    """Кэшированная генерация комментариев"""
    return _model.generate_comments(prompt=prompt, **generation_params)

def extract_video_info(url):
    """Извлечение информации о видео из URL с использованием YouTube Transcript API"""
    try:
        # Извлекаем video_id из URL
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url and "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        else:
            st.error("Неверный формат URL. Используйте URL вида https://www.youtube.com/watch?v=ID_VIDEO")
            return None
        
        # Получаем базовое превью изображение
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        
        # Получаем oembed данные для базовой информации о видео
        import requests
        import json
        
        try:
            # Метод 1: Получаем данные через oEmbed API
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url)
            
            if response.status_code == 200:
                oembed_data = response.json()
                title = oembed_data.get('title', 'Название недоступно')
                author = oembed_data.get('author_name', 'Автор недоступен')
                st.success("Успешно получена информация о видео через oEmbed API")
            else:
                title = f"YouTube видео (ID: {video_id})"
                author = "YouTube Creator"
                st.warning("Не удалось получить данные через oEmbed API, используем базовую информацию")
        except Exception as e:
            st.warning(f"Ошибка при получении данных через oEmbed API: {str(e)}")
            title = f"YouTube видео (ID: {video_id})"
            author = "YouTube Creator"
        
        # Получаем транскрипцию видео
        transcript_text = ""
        try:
            # Получаем доступные транскрипции
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Приоритет для английских субтитров
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # Если английские недоступны, берем первый доступный и переводим
                transcript = transcript_list.find_transcript(['en-US', 'en-GB'])
            
            # Если всё еще не нашли, берем любой и переводим на английский
            if not transcript:
                transcript = transcript_list[0].translate('en')
            
            # Получаем полный текст
            transcript_data = transcript.fetch()
            
            # Объединяем текст всех сегментов
            transcript_text = " ".join([item['text'] for item in transcript_data])
            
            # Ограничиваем длину для UI
            transcript_display = transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text
            
            st.success(f"Успешно получена транскрипция видео ({len(transcript_text)} символов)")
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            st.warning(f"Транскрипция недоступна для этого видео: {str(e)}")
            transcript_text = ""
        except Exception as e:
            st.warning(f"Ошибка при получении транскрипции: {str(e)}")
            transcript_text = ""
        
        # Создаем объект с информацией о видео
        video_info = {
            "title": title,
            "author": author,
            "description": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
            "full_transcript": transcript_text,
            "thumbnail_url": thumbnail_url,
            "views": 0,  # Нет доступа без API ключа
            "length": 0,  # Нет доступа без API ключа
            "is_english": True,  # Предполагаем, что транскрипция на английском или переведена
            "video_id": video_id
        }
        
        return video_info
        
    except Exception as e:
        st.error(f"Ошибка при обработке URL: {str(e)}")
        st.info("Попробуйте другое видео или используйте режим ручного ввода информации.")
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

def extract_keywords(text, max_keywords=5, min_word_length=4):
    """Извлечение ключевых слов из текста"""
    if not text:
        return []
    
    # Приводим к нижнему регистру и удаляем специальные символы
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Список стоп-слов (часто встречающиеся слова, не несущие смысловой нагрузки)
    stop_words = {'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on',
                  'with', 'as', 'this', 'by', 'are', 'or', 'be', 'if', 'from', 'an',
                  'but', 'not', 'they', 'what', 'all', 'when', 'who', 'will', 'more',
                  'about', 'which', 'can', 'their', 'has', 'would', 'should', 'been',
                  'could', 'you', 'your', 'them', 'were', 'was', 'how', 'than', 'then',
                  'now', 'very', 'just', 'have', 'some', 'like', 'also', 'only', 'its',
                  'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third'}
    
    # Разбиваем на слова
    words = clean_text.split()
    
    # Фильтруем слова
    filtered_words = [word for word in words 
                      if word not in stop_words and len(word) >= min_word_length]
    
    # Подсчет частоты слов
    word_counts = Counter(filtered_words)
    
    # Получаем наиболее часто встречающиеся слова
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords

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
        1. Введите URL-адрес видео YouTube или используйте ручной ввод
        2. При вводе URL будет автоматически получена транскрипция видео
        3. Настройте параметры генерации
        4. Нажмите 'Сгенерировать комментарии'
        
        **Особенности:**
        - Приложение получает транскрипцию видео для создания более релевантных комментариев
        - Автоматически извлекаются ключевые слова из видео
        - Доступен ручной ввод для видео без субтитров
        
        **Оптимальные настройки:**
        - Для быстрой генерации: Beam Search с beam size 2-3, макс. длина 40-60
        - Для разнообразных комментариев: Sampling с temperature 0.7-0.9
        - Отключите потоковую генерацию для ускорения работы
        """)
        
        # Добавляем информацию о возможных проблемах
        st.subheader("Возможные проблемы")
        st.markdown("""
        - Не все видео имеют доступные транскрипции
        - Генерация может быть медленной в облачной среде
        - Если комментарии не генерируются, попробуйте уменьшить параметры или перезапустить приложение
        """)
    
    # Переключатель режима ввода
    input_mode = st.radio(
        "Выберите режим ввода:",
        options=["Ссылка на YouTube видео", "Ручной ввод информации"],
        index=0,
        horizontal=True
    )
    
    video_info = None
    
    if input_mode == "Ссылка на YouTube видео":
        # Ввод URL видео
        url = st.text_input("Введите URL видео YouTube:", help="Например: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        if url:
            with st.spinner("Извлечение информации о видео..."):
                video_info = extract_video_info_cached(url)
    else:
        # Ручной ввод информации о видео
        st.subheader("Введите информацию о видео:")
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Название видео:", value="Interesting Machine Learning Video")
            author = st.text_input("Автор видео:", value="AI Channel")
        
        with col2:
            views = st.number_input("Количество просмотров:", min_value=0, value=10000)
            length = st.number_input("Длительность (в секундах):", min_value=0, value=600)
        
        description = st.text_area("Описание видео:", value="This video discusses the latest advancements in artificial intelligence and machine learning.")
        
        # Добавляем поле для транскрипции
        transcript = st.text_area("Транскрипция видео (необязательно):", value="", height=150, 
                               help="Введите часть транскрипции видео для более релевантных комментариев")
        
        # Создание объекта с информацией о видео
        video_info = {
            "title": title,
            "author": author,
            "description": description,
            "full_transcript": transcript,
            "thumbnail_url": "https://img.youtube.com/vi/placeholder/0.jpg",
            "views": views,
            "length": length,
            "is_english": True,
            "video_id": "manual_input"
        }
    
    # Выбор метода генерации
    generation_method = st.radio(
        "Метод генерации:",
        options=["Sampling", "Beam Search"],
        index=0,
        horizontal=True
    )
    
    # Параметры генерации - первый ряд
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_comments = st.slider("Количество комментариев:", min_value=1, max_value=10, value=3, 
                               help="Меньшее количество позволяет быстрее генерировать комментарии")
    
    with col2:
        if generation_method == "Sampling":
            temperature = st.slider("Temperature:", min_value=0.1, max_value=1.5, value=0.8, step=0.1,
                                  help="Выше = более разнообразные результаты. Рекомендуется 0.7-0.9")
        else:
            beam_size = st.slider("Beam Size:", min_value=2, max_value=5, value=3, step=1,
                                help="Больше = качественнее, но медленнее. Рекомендуется 2-3 для быстрой генерации")
    
    with col3:
        if generation_method == "Sampling":
            top_p = st.slider("Top-p:", min_value=0.1, max_value=1.0, value=0.92, step=0.02,
                             help="Nucleus sampling: меньше = консервативнее. Рекомендуется 0.90-0.95")
        else:
            length_penalty = st.slider("Length Penalty:", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                      help="Выше = более длинные комментарии. Рекомендуется 0.8-1.2")
    
    # Параметры генерации - второй ряд
    col4, col5, col6 = st.columns(3)
    
    with col4:
        max_length = st.slider("Максимальная длина:", min_value=30, max_value=150, value=60, step=10,
                              help="Меньшая длина ускоряет генерацию. Для комментариев обычно хватает 50-80 токенов")
    
    with col5:
        pass
    
    with col6:
        streaming_mode = st.checkbox("Потоковая генерация (слово за словом)", value=False,
                                    help="Отключение ускоряет генерацию комментариев")
    
    # Сгенерировать комментарии
    if st.button("Сгенерировать комментарии"):
        if video_info:
            if video_info["is_english"]:
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
                
                # Отображение транскрипции, если она доступна
                if video_info.get("full_transcript"):
                    with st.expander("Транскрипция видео"):
                        transcript_text = video_info["full_transcript"]
                        st.markdown(transcript_text[:2000] + "..." if len(transcript_text) > 2000 else transcript_text)
                
                # Загрузка модели
                try:
                    with st.spinner("Загрузка модели..."):
                        model = load_model()
                    
                    # Подготовка промпта на основе видео и транскрипции
                    if not video_info.get("title") or video_info["title"] == "Название недоступно":
                        # Если нет названия, используем базовый промпт
                        prompt = f"<BOS>This YouTube video is really interesting"
                    else:
                        # Создаем промпт на основе названия
                        clean_title = video_info["title"].replace("<", "").replace(">", "")
                        prompt = f"<BOS>This video about {clean_title} is really interesting"
                    
                    # Добавляем автора, если доступен
                    if video_info.get("author") and video_info["author"] != "Автор недоступен":
                        prompt += f". {video_info['author']} makes great content"
                    
                    # Добавляем ключевые слова из транскрипции для контекста, если доступна
                    if video_info.get("full_transcript"):
                        # Извлекаем ключевые слова
                        keywords = extract_keywords(video_info["full_transcript"], max_keywords=7)
                        
                        if keywords:
                            # Формируем строку с ключевыми словами
                            keywords_str = ", ".join(keywords)
                            prompt += f". The video covers topics like {keywords_str}"
                        else:
                            # Берем начало транскрипции для контекста (не более 100 символов)
                            transcript_start = video_info["full_transcript"][:100].replace("<", "").replace(">", "")
                            prompt += f". In the video they talk about: '{transcript_start}...'"
                    
                    # Отладочная информация о промпте
                    st.info(f"Используемый промпт: {prompt}")
                    
                    # Формирование параметров генерации
                    generation_params = {
                        "max_length": max_length,
                    }
                    
                    if generation_method == "Sampling":
                        generation_params.update({
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": top_p,
                        })
                    else:  # Beam Search
                        generation_params.update({
                            "do_sample": False,
                            "num_beams": beam_size,
                            "length_penalty": length_penalty,
                        })
                    
                    # Вывод комментариев
                    st.subheader("Сгенерированные комментарии:")
                    
                    # Массив для хранения комментариев
                    comments = []
                    
                    # Генерация комментариев
                    for i in range(num_comments):
                        st.markdown(f"### Комментарий {i + 1}:")
                        
                        if streaming_mode:
                            # Потоковая генерация с выводом по токенам
                            comment_placeholder = st.empty()
                            current_comment = ""
                            
                            # Функция обратного вызова для обновления UI
                            def update_ui(token):
                                nonlocal current_comment
                                current_comment += token
                                comment_placeholder.markdown(current_comment + "▌")
                            
                            # Генерация комментария с потоковым выводом
                            full_comment = model.generate_comments_stream(
                                prompt=prompt,
                                callback=update_ui,
                                **generation_params
                            )
                            
                            # Обновление финального комментария без курсора
                            full_comment = full_comment.replace("<BOS>", "").strip()
                            comment_placeholder.markdown(full_comment)
                            comments.append(full_comment)
                        else:
                            # Обычная генерация (с кэшированием)
                            with st.spinner(f"Генерация комментария {i + 1}..."):
                                try:
                                    # Добавляем небольшое изменение к промпту для каждого комментария,
                                    # чтобы избежать одинаковых результатов при кэшировании
                                    modified_prompt = prompt + f" comment{i}"
                                    
                                    # Если генерация не работает, пробуем более простой промпт
                                    if i > 0 and not comments:
                                        modified_prompt = "<BOS>This scientific YouTube video"
                                        # Убрано сообщение о простом промпте
                                    
                                    result = generate_comments_cached(
                                        model,
                                        prompt=modified_prompt,
                                        generation_params=generation_params
                                    )
                                    
                                    if result and len(result) > 0:
                                        comment = result[0].replace("<BOS>", "").strip()
                                        st.markdown(comment)
                                        comments.append(comment)
                                    else:
                                        st.error("Не удалось сгенерировать комментарий")
                                except Exception as e:
                                    st.error(f"Ошибка при генерации комментария: {str(e)}")
                                    # Пробуем с более простым промптом
                                    try:
                                        simple_prompt = "<BOS>This scientific YouTube video"
                                        # Убрано сообщение о простом промпте
                                        result = model.generate_comments(
                                            prompt=simple_prompt,
                                            num_return_sequences=1,
                                            max_length=50,
                                            temperature=0.7,
                                            do_sample=True
                                        )
                                        if result and len(result) > 0:
                                            comment = result[0].replace("<BOS>", "").strip()
                                            st.markdown(comment)
                                            comments.append(comment)
                                    except Exception as e2:
                                        st.error(f"Не удалось сгенерировать комментарий даже с простым промптом: {str(e2)}")
                        
                        st.markdown("---")
                    
                    # Блок "Ответы на комментарии" удален
                
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