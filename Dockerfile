FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов проекта
COPY . .

# Создание директории для данных
RUN mkdir -p data/processed data/prepared model_output

# Обучение модели
RUN python -c "from train_model import CommentGenerationModel; \
    model = CommentGenerationModel(model_name='gpt2-large', output_dir='./model_output'); \
    tokenizer, gpt2_model = model.load_tokenizer_and_model(); \
    train_path, test_path = model.prepare_dataset(csv_path='data/processed/train.csv', \
    output_dir='data/prepared', column_name='clean_text'); \
    train_dataset = model.load_dataset(train_path); \
    test_dataset = model.load_dataset(test_path); \
    model.train(train_dataset, test_dataset)"

# Установка порта
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 