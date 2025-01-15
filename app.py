from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Intentar cargar el modelo y el tokenizador desde la carpeta donde los guardaste
try:
    model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
    tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_model")
    print("Modelo y tokenizador cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o tokenizador: {e}")
    model = None
    tokenizer = None

# Página principal con el formulario HTML
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para manejar las solicitudes del chat
@app.route('/chat', methods=['POST'])
def chat():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Modelo o tokenizador no cargados correctamente.'}), 500

    # Cambiar 'request.form' a 'request.json' para obtener datos en formato JSON
    user_input = request.json.get('message')  # Usamos request.json.get('message')
    
    if not user_input:  # Si no se recibe ningún mensaje
        return jsonify({'error': 'No se recibió el mensaje'}), 400
    
    # Preparar la entrada para el modelo
    input_text = f"responde a la siguiente pregunta: {user_input}"

    # Tokenizar la entrada
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Generar la respuesta con los parámetros ajustados para mejorar la calidad de las respuestas
    outputs = model.generate(
        inputs,
        max_length=150,          # Longitud máxima de la respuesta
        num_beams=10,            # Aumentar la búsqueda de haz
        no_repeat_ngram_size=3,  # Evitar que se repitan frases
        top_k=50,                # Limitar el número de opciones de palabras
        top_p=0.95,              # Control de probabilidad acumulada
        early_stopping=True      # Parar si ya se genera una respuesta completa
    )

    # Decodificar la respuesta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Imprimir la respuesta generada para depuración
    print(f"Respuesta generada: {response}")
    
    # Devolver la respuesta como JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)