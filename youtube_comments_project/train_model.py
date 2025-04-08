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
        self.compiled_model = None  # Для хранения скомпилированной модели
        self.inference_model = None  # Для хранения оптимизированной для инференса модели
        
        # Создаем директорию для выходных данных, если ее нет
        os.makedirs(output_dir, exist_ok=True)
    
    def load_tokenizer_and_model(self):
        """Загрузка токенизатора и модели с оптимизациями"""
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
    
    def optimize_model_for_inference(self):
        """Оптимизация модели только для инференса (не для обучения)"""
        if self.model is None:
            self.load_tokenizer_and_model()
        
        # Создаем копию модели для инференса
        inference_model = self.model
        
        # Применение квантизации для ускорения инференса
        try:
            logger.info("Применение квантизации int8...")
            inference_model = torch.quantization.quantize_dynamic(
                inference_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Квантизация успешно применена")
        except Exception as e:
            logger.warning(f"Не удалось применить квантизацию: {str(e)}")
        
        # Попытка использовать torch.compile для ускорения
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            try:
                logger.info("Компиляция модели с torch.compile...")
                self.compiled_model = torch.compile(inference_model)
                logger.info("Модель успешно скомпилирована")
            except Exception as e:
                logger.warning(f"Не удалось скомпилировать модель: {str(e)}")
                self.compiled_model = None
        else:
            logger.warning("torch.compile недоступен (требуется PyTorch >= 2.0.0)")
            self.compiled_model = None
        
        return inference_model
    
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
                         temperature=1.0, top_k=50, top_p=0.95, do_sample=True,
                         num_beams=None, length_penalty=1.0):
        """Генерация комментариев с оптимизациями"""
        try:
            # Если модель не загружена, загружаем ее
            if self.model is None:
                self.load_tokenizer_and_model()
            
            # Если промпт не задан, используем начало комментария
            if prompt is None or prompt == "":
                prompt = "<BOS>"
            
            # Проверка и очистка промпта
            if isinstance(prompt, str):
                # Удаляем специальные символы, кроме тегов <BOS> и <EOS>
                clean_prompt = prompt
                if not clean_prompt.startswith("<BOS>"):
                    clean_prompt = "<BOS>" + clean_prompt
            else:
                logger.warning(f"Получен некорректный промпт типа {type(prompt)}, используем значение по умолчанию")
                clean_prompt = "<BOS>"
            
            # Максимальная длина не должна быть слишком большой в облачной среде
            safe_max_length = min(max_length, 150)  # Ограничиваем максимальную длину
            
            # Переводим модель в режим генерации
            self.model.eval()
            
            # Оптимизируем модель для инференса, если это еще не сделано
            if not hasattr(self, 'inference_model') or self.inference_model is None:
                try:
                    self.inference_model = self.optimize_model_for_inference()
                except Exception as e:
                    logger.warning(f"Не удалось оптимизировать модель: {str(e)}. Используем обычную модель.")
                    self.inference_model = self.model
            
            # Токенизация промпта
            try:
                inputs = self.tokenizer(clean_prompt, return_tensors="pt", add_special_tokens=False)
            except Exception as e:
                logger.error(f"Ошибка при токенизации промпта: {str(e)}")
                # Пробуем простейший промпт
                inputs = self.tokenizer("<BOS>", return_tensors="pt", add_special_tokens=False)
            
            # Настройка параметров генерации
            generation_config = {
                "max_length": safe_max_length,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Добавляем параметры в зависимости от метода генерации
            if do_sample:
                generation_config.update({
                    "do_sample": True,
                    "temperature": min(max(temperature, 0.1), 1.5),  # Ограничиваем диапазон
                    "top_k": min(max(top_k, 1), 100),                # Ограничиваем диапазон
                    "top_p": min(max(top_p, 0.1), 1.0),              # Ограничиваем диапазон
                })
            elif num_beams:
                generation_config.update({
                    "do_sample": False,
                    "num_beams": min(max(num_beams, 1), 5),            # Ограничиваем для экономии памяти
                    "length_penalty": min(max(length_penalty, 0.1), 2.0), # Ограничиваем диапазон
                })
            
            # Генерация текста с отключенным градиентом для экономии памяти
            with torch.no_grad():
                # Используем скомпилированную модель, если доступна
                model_to_use = self.compiled_model if self.compiled_model is not None else self.inference_model
                
                # Проверяем, что модель и входные данные на одном устройстве
                try:
                    outputs = model_to_use.generate(
                        inputs.input_ids,
                        **generation_config
                    )
                except Exception as e:
                    logger.error(f"Ошибка при генерации: {str(e)}")
                    return ["Ошибка генерации. Пожалуйста, используйте более короткий промпт или уменьшите параметры генерации."]
            
            # Декодирование результатов
            comments = []
            for output in outputs:
                try:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    if text.strip():  # Проверяем, что текст не пустой
                        comments.append(text)
                    else:
                        comments.append("Комментарий не сгенерирован. Попробуйте изменить параметры.")
                except Exception as e:
                    logger.error(f"Ошибка при декодировании: {str(e)}")
                    comments.append("Ошибка декодирования.")
            
            # Если не удалось ничего сгенерировать, добавляем заглушку
            if not comments:
                comments = ["Комментарии не сгенерированы. Попробуйте изменить настройки генерации."]
            
            return comments
        except Exception as e:
            logger.error(f"Ошибка при генерации комментариев: {str(e)}")
            return [f"Произошла ошибка при генерации: {str(e)}"]
    
    def generate_comments_stream(self, prompt=None, max_length=100, 
                               temperature=1.0, top_k=50, top_p=0.95, 
                               do_sample=True, num_beams=None, 
                               length_penalty=1.0, callback=None):
        """Генерация комментариев с потоковым выводом по токенам с оптимизациями"""
        try:
            # Если модель не загружена, загружаем ее
            if self.model is None:
                self.load_tokenizer_and_model()
            
            # Если промпт не задан, используем начало комментария
            if prompt is None or prompt == "":
                prompt = "<BOS>"
            
            # Переводим модель в режим генерации
            self.model.eval()
            
            # Оптимизируем модель для инференса, если это еще не сделано
            if not hasattr(self, 'inference_model') or self.inference_model is None:
                self.inference_model = self.optimize_model_for_inference()
            
            # Токенизация промпта
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_length = inputs.input_ids.shape[1]
            
            # Настройка параметров генерации
            generation_config = {
                "max_length": input_length + max_length,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Добавляем параметры в зависимости от метода генерации
            if do_sample:
                generation_config.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                })
            elif num_beams:
                generation_config.update({
                    "do_sample": False,
                    "num_beams": num_beams,
                    "length_penalty": length_penalty,
                })
            
            # Начальный токен из входных данных
            generated = inputs.input_ids
            
            # Функция для обработки каждого нового токена
            def process_token(token):
                token_text = self.tokenizer.decode(token, skip_special_tokens=True)
                if callback and token_text.strip():
                    callback(token_text)
                return token_text
            
            # Полный сгенерированный текст
            full_text = ""
            
            # Используем скомпилированную модель, если доступна
            model_to_use = self.compiled_model if self.compiled_model is not None else self.inference_model
            
            # Оптимизируем: генерируем весь текст сразу, если не требуется потоковый вывод
            if callback is None:
                with torch.no_grad():
                    outputs = model_to_use.generate(
                        generated,
                        **generation_config
                    )
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return full_text
            
            # Генерация по одному токену за раз
            with torch.no_grad():
                for _ in range(max_length):
                    outputs = model_to_use(generated)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Применяем температуру
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Применяем top_k
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("Inf")
                    
                    # Применяем top_p (nucleus sampling)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Удаляем токены с вероятностью выше порога
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[0, indices_to_remove] = -float("Inf")
                    
                    # Выбираем следующий токен
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Добавляем токен к сгенерированной последовательности
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Обрабатываем новый токен
                    token_text = process_token(next_token.squeeze().item())
                    full_text += token_text
                    
                    # Проверка на конец последовательности
                    if next_token.squeeze().item() == self.tokenizer.eos_token_id:
                        break
            
            return full_text
        except Exception as e:
            logger.error(f"Ошибка при потоковой генерации комментариев: {str(e)}")
            if callback:
                callback(f" [Ошибка: {str(e)}]")
            return f"Произошла ошибка при генерации: {str(e)}"


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
    
    # Генерация комментариев
    comments = model.generate_comments(prompt="<BOS>This video", num_return_sequences=3)
    print("\nСгенерированные комментарии:")
    for comment in comments:
        print(f"- {comment}") 