from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Cargar datos de entrenamiento
with open("train_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Preparar el conjunto de datos
input_texts = [item['input_text'] for item in data]
target_texts = [item['target_text'] for item in data]
dataset = {"input_text": input_texts, "target_text": target_texts}

# Convertir los datos a formato compatible con Hugging Face
train_dataset = Dataset.from_dict(dataset)

# Crear un conjunto de datos de evaluación (por ejemplo, usando los primeros 100 ejemplos de los datos de entrenamiento)
eval_dataset = train_dataset.select(range(100))

# Cargar el modelo y el tokenizador
model_name = "t5-small"  # O usa "t5-base" si prefieres un modelo más grande
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tokenizar los datos
def tokenize_data(examples):
    inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = outputs['input_ids']
    return inputs

train_dataset = train_dataset.map(tokenize_data, batched=True)
eval_dataset = eval_dataset.map(tokenize_data, batched=True)

# Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",           # Directorio donde se guardará el modelo entrenado
    evaluation_strategy="epoch",      # Evaluar cada época
    learning_rate=2e-5,               # Tasa de aprendizaje
    per_device_train_batch_size=8,    # Tamaño del batch
    per_device_eval_batch_size=8,     # Tamaño del batch para evaluación
    num_train_epochs=3,               # Número de épocas
    weight_decay=0.01,                # Decaimiento de peso
    logging_dir="./logs",             # Directorio de logs
)

# Configurar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Agregar el conjunto de datos de evaluación
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo fine-tuneado
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")