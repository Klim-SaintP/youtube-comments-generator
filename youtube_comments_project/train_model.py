import os
import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommentGenerationModel:
    def __init__(self, model_name="gpt2-large", output_dir="./model_output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
        # Создаем директорию для выходных данных, если ее нет
        os.makedirs(output_dir, exist_ok=True)
    
    def load_tokenizer_and_model(self):
        """Загрузка токенизатора и модели"""
        logger.info(f"Загрузка токенизатора и модели {self.model_name}...")
        
        # Загрузка токенизатора
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Добавление специальных токенов
        special_tokens = {
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
        }
        
        # Обновление токенизатора
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Загрузка модели
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Токенизатор и модель загружены.")
        
        return self.tokenizer, self.model
    
    def prepare_dataset(self, csv_path, output_dir, column_name='clean_text', 
                       block_size=128, test_size=0.1):
        """Подготовка датасета для обучения"""
        logger.info(f"Подготовка датасета из {csv_path}...")
        
        # Если токенизатора нет, загружаем его
        if self.tokenizer is None:
            self.load_tokenizer_and_model()
        
        # Загрузка данных
        df = pd.read_csv(csv_path)
        
        # Проверка наличия нужной колонки
        if column_name not in df.columns:
            raise ValueError(f"Колонка {column_name} не найдена в данных.")
        
        # Подготовка данных
        texts = df[column_name].dropna().tolist()
        
        # Создаем директории для тренировочных и тестовых данных
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Разделение на тренировочную и тестовую выборки
        train_size = int(len(texts) * (1 - test_size))
        train_texts = texts[:train_size]
        test_texts = texts[train_size:]
        
        # Создание файлов для обучения
        train_path = os.path.join(train_dir, "train.txt")
        test_path = os.path.join(test_dir, "test.txt")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for text in train_texts:
                # Добавляем специальные токены в начало и конец
                f.write(f"<BOS>{text}<EOS>\n")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            for text in test_texts:
                f.write(f"<BOS>{text}<EOS>\n")
        
        logger.info(f"Создано {len(train_texts)} тренировочных и {len(test_texts)} тестовых примеров.")
        
        return train_path, test_path
    
    def load_dataset(self, file_path, block_size=128):
        """Загрузка датасета для обучения"""
        return TextDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            block_size=block_size,
        )
    
    def train(self, train_dataset, eval_dataset=None, num_train_epochs=3, 
             per_device_train_batch_size=4, gradient_accumulation_steps=8):
        """Обучение модели"""
        logger.info("Начало обучения модели...")
        
        # Настройка параметров обучения
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            warmup_steps=500,
        )
        
        # Подготовка коллатора данных
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Авторегрессионная языковая модель, а не masked LM
        )
        
        # Инициализация тренера
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Обучение модели
        trainer.train()
        
        # Сохранение модели
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Модель обучена и сохранена в {self.output_dir}")
        
        return trainer
    
    def generate_comments(self, prompt=None, max_length=100, num_return_sequences=1, 
                         temperature=1.0, top_k=50, top_p=0.95, do_sample=True):
        """Генерация комментариев"""
        # Если модель не загружена, загружаем ее
        if self.model is None:
            self.load_tokenizer_and_model()
        
        # Если промпт не задан, используем начало комментария
        if prompt is None or prompt == "":
            prompt = "<BOS>"
        
        # Токенизация промпта
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        
        # Генерация текста
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Декодирование результатов
        comments = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            comments.append(text)
        
        return comments


if __name__ == "__main__":
    # Инициализация модели
    model = CommentGenerationModel(model_name="gpt2-large", output_dir="./model_output")
    
    # Загрузка токенизатора и модели
    tokenizer, gpt2_model = model.load_tokenizer_and_model()
    
    # Подготовка датасета
    train_path, test_path = model.prepare_dataset(
        csv_path="data/processed/train.csv", 
        output_dir="data/prepared",
        column_name="clean_text",
    )
    
    # Загрузка датасетов
    train_dataset = model.load_dataset(train_path)
    eval_dataset = model.load_dataset(test_path)
    
    # Обучение модели
    trainer = model.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
    )
    
    # Генерация примеров комментариев
    comments = model.generate_comments(
        prompt="<BOS>This video is",
        num_return_sequences=3,
        temperature=0.8,
    )
    
    print("\nПримеры сгенерированных комментариев:")
    for i, comment in enumerate(comments, 1):
        print(f"{i}. {comment}") 