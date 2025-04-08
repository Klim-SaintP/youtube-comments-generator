import os
from train_model import CommentGenerationModel

def train_if_needed():
    if not os.path.exists("./model_output/pytorch_model.bin"):
        print("Обучение модели...")
        # Создание директорий, если их не существует
        os.makedirs("data/prepared/train", exist_ok=True)
        os.makedirs("data/prepared/test", exist_ok=True)
        os.makedirs("model_output", exist_ok=True)
        
        model = CommentGenerationModel(model_name='gpt2-large', output_dir='./model_output')
        model.load_tokenizer_and_model()
        
        train_path, test_path = model.prepare_dataset(
            csv_path='data/processed/train.csv', 
            output_dir='data/prepared', 
            column_name='clean_text'
        )
        
        train_dataset = model.load_dataset(train_path)
        test_dataset = model.load_dataset(test_path)
        
        model.train(
            train_dataset, 
            test_dataset, 
            num_train_epochs=3, 
            per_device_train_batch_size=2, 
            gradient_accumulation_steps=8
        )
        
        print("Модель обучена и сохранена.")
    else:
        print("Модель уже обучена, пропускаем обучение.")

if __name__ == "__main__":
    train_if_needed() 