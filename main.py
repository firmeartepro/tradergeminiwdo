# Importa bibliotecas essenciais
import os
import json
import time
from flask import Flask, request, jsonify
from supabase import create_client, Client
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from keras.models import load_model

# --- Configurações Globais ---
# A aplicação Flask
app = Flask(__name__)

# Variáveis globais para o modelo e serviços
model = None
db_client = None

# Configurações do modelo
MODEL_PATH = 'ia_wdo_v1_ticks.h5'
TIME_STEP = 300
SCALER_RANGE = (0, 1)

# --- Funções de Inicialização ---

def inicializar_modelo():
    """Carrega o modelo Keras/TensorFlow a partir do arquivo H5."""
    global model
    try:
        # Garante que o TensorFlow só use a CPU no ambiente serverless
        # Para evitar problemas de compatibilidade com GPU
        tf.config.set_visible_devices([], 'GPU')
        
        # Carrega o modelo
        model = load_model(MODEL_PATH)
        print("Modelo de IA carregado com sucesso.")
        return True
    except Exception as e:
        print(f"ERRO ao carregar o modelo de IA: {e}")
        return False

def inicializar_servicos():
    """Inicializa a conexão com o Supabase."""
    global db_client
    try:
        # As variáveis de ambiente devem ser configuradas no painel do Deta Space
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            print("AVISO: Variáveis SUPABASE_URL ou SUPABASE_KEY não encontradas. Logs desativados.")
            return False

        db_client = create_client(url, key)
        print("Conexão Supabase inicializada.")
        return True
    except Exception as e:
        print(f"ERRO ao conectar ao Supabase: {e}")
        return False

def registrar_sinal(data, prediction_value, signal_time):
    """Salva o log do sinal no Supabase."""
    if not db_client:
        print("Supabase não conectado. Log não realizado.")
        return

    try:
        data_to_log = {
            "timestamp": int(time.time()),
            "sinal_mt5": data.get("Signal"),
            "prediction_ia": prediction_value,
            "candle_time": signal_time,
            "data_recebida": data 
        }
        
        # O nome da tabela deve ser ajustado para a sua configuração do Supabase
        # Usaremos 'sinais_wdo' como padrão
        response = db_client.table("sinais_wdo").insert(data_to_log).execute()
        print(f"Sinal registrado com sucesso: {response.data}")

    except Exception as e:
        print(f"ERRO ao registrar sinal no Supabase: {e}")

# --- Endpoint da API ---

@app.route('/', methods=['GET'])
def health_check():
    """Verificação de saúde simples. Deve retornar OK."""
    if model and db_client:
        return jsonify({"status": "OK", "message": "API pronta e conectada ao Supabase."}), 200
    elif model:
        return jsonify({"status": "OK", "message": "API pronta, modelo carregado, Supabase desconectado."}), 200
    else:
        return jsonify({"status": "ERROR", "message": "API offline ou modelo não carregado."}), 500

@app.route('/', methods=['POST'])
def processar_dados():
    """Recebe os dados do MT5 e retorna a previsão da IA."""
    global model

    if not model:
        return jsonify({"Error": "Modelo de IA não carregado na API."}), 503

    try:
        # 1. Recebe os dados do MT5 (JSON)
        dados = request.get_json(force=True)
        
        # 2. Extrai e valida a lista de preços
        precos_str = dados.get('Prices')
        if not precos_str:
            return jsonify({"Error": "Campo 'Prices' não encontrado nos dados."}), 400

        # Converte a string de preços (separada por vírgulas) para uma lista de floats
        precos = [float(p) for p in precos_str.split(',') if p]

        # 3. Verifica o tamanho da série temporal
        if len(precos) < TIME_STEP:
            return jsonify({"Error": f"Série temporal incompleta. Esperado: {TIME_STEP}, Recebido: {len(precos)}"}), 400

        # Garante que o input tenha o tamanho exato necessário pelo modelo (TIME_STEP)
        # Pega os últimos TIME_STEP preços
        precos_a_processar = np.array(precos[-TIME_STEP:]).reshape(-1, 1)

        # 4. Normalização (Escalonamento)
        # Simula o escalonamento usado no treinamento do modelo (Max-Min Scaler)
        min_val = np.min(precos_a_processar)
        max_val = np.max(precos_a_processar)
        
        if max_val == min_val:
            # Evita divisão por zero se todos os preços forem iguais
            dados_normalizados = np.zeros_like(precos_a_processar)
        else:
            dados_normalizados = (precos_a_processar - min_val) / (max_val - min_val)

        # 5. Formatação para a IA (Adiciona a dimensão de batch)
        # Formato esperado: (1, TIME_STEP, 1)
        dados_formatados = dados_normalizados.reshape(1, TIME_STEP, 1)

        # 6. Previsão da IA
        prediction = model.predict(dados_formatados)
        # A previsão é um valor entre 0 e 1.
        prediction_value = prediction[0][0]

        # 7. Tradução do Sinal (Lógica da Aplicação)
        # O modelo prevê a probabilidade de um movimento.
        if prediction_value > 0.6:  # Alta confiança na alta (COMPRA)
            sinal_ia = "BUY"
        elif prediction_value < 0.4: # Alta confiança na baixa (VENDA)
            sinal_ia = "SELL"
        else:
            sinal_ia = "HOLD" # Indefinido

        # 8. Registro e Resposta
        signal_time = dados.get('CandleTime', 'N/A')
        registrar_sinal(dados, float(prediction_value), signal_time)
        
        # Retorna o sinal e a pontuação de confiança da IA
        return jsonify({
            "Signal": sinal_ia,
            "Confidence": float(prediction_value) # Garante que seja float serializável
        }), 200

    except Exception as e:
        # Em caso de qualquer erro interno
        print(f"Falha no processamento: {e}")
        return jsonify({"Error": f"Erro interno ao processar a requisição: {str(e)}"}), 500

# --- Inicialização da Aplicação ---

# O Deta Space executa o main.py.
# Esta parte é crucial para carregar o modelo e conectar o Supabase ANTES de servir requisições.
if __name__ != '__main__':
    # O Deta executa esta parte apenas uma vez durante a inicialização do container
    inicializar_modelo()
    inicializar_servicos()

# No Deta, a variável global 'app' é detectada e servida automaticamente.
# Não precisamos de 'if __name__ == "__main__"' para rodar o servidor.


