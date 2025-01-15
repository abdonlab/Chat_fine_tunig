from transformers import T5ForConditionalGeneration, T5Tokenizer

# Cargar el modelo y el tokenizador desde la carpeta donde los guardaste
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_model")

# Ejemplo de entrada
user_input = "¿Cuáles son los requisitos para ingresar?"

# Preparar la entrada para el modelo
input_text = f"responde a la siguiente pregunta: {user_input}"

# Tokenizar la entrada
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

# Generar la respuesta
outputs = model.generate(inputs["input_ids"], max_length=100, num_beams=4, early_stopping=True)

# Decodificar la respuesta
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Mostrar la respuesta generada
print("Respuesta del modelo:", response)