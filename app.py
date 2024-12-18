from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregar o modelo DialoGPT
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configuração do Flask
app = Flask(__name__)

# Função para gerar resposta do chatbot
def chat_with_dialoggpt(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    response = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    chat_output = tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_output

# Endpoint da API
@app.route('/api', methods=['POST'])
def api():
    if request.is_json:
        data = request.get_json()
        message = data.get('message', '')
        response = chat_with_dialoggpt(message)
        return jsonify({"message": response}), 200
    else:
        return jsonify({"error": "A requisição precisa ser no formato JSON"}), 400

# Inicializar o servidor no Render (opcional para testes locais)
if __name__ == '__main__':
    app.run(debug=True)
