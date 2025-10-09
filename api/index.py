# api/index.py
import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_ta as ta
from supabase import create_client, Client

app = Flask(__name__)

# --- CONFIGURAÇÕES SUPABASE (Lidas de Secrets do Vercel) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = "wdo_trade_logs"

# --- CONFIGURAÇÕES DO MODELO ---
THRESHOLD_COMPRA = 0.60
COLUNAS_INDICADORES = [
    'RSI_14', 'MACDh_12_26_9', 'ADX_14', 'CCI_14_0.015', 
    'BBP_5_2.0', 'BBL_5_2.0', 'ATR_14'
]

model_ia = None
supabase: Client = None

def inicializar_servicos():
    """Carrega o modelo e inicializa o cliente Supabase. Executado no cold start."""
    global model_ia, supabase
    
    # Carrega Modelo
    try:
        # Usando compile=False para evitar erros de compatibilidade e acelerar o carregamento
        print("Iniciando carregamento do modelo ia_wdo_v1_ticks.h5...")
        model_ia = tf.keras.models.load_model('ia_wdo_v1_ticks.h5', compile=False) 
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        
    # Inicializa Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            print("Inicializando cliente Supabase...")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("Supabase inicializado.")
        except Exception as e:
            print(f"ERRO ao inicializar Supabase: {e}")

# O modelo e o Supabase são carregados na inicialização do worker (cold start)
inicializar_servicos()


def preparar_input_ia(historico: pd.DataFrame):
    """Calcula os 7 indicadores e retorna o array para previsão e os valores para log."""
    df = historico[['open', 'high', 'low', 'close', 'tick_volume']].copy()
    
    # Cálculo dos Indicadores (Garantindo que a coluna 'close' exista para o pandas_ta)
    df.ta.bbands(close='close', length=5, append=True) 
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, slow=9, append=True) # ADX precisa de um valor para o 'slow'
    df.ta.cci(length=14, append=True)

    # Renomeia o CCI para coincidir com o que o modelo espera
    df = df.rename(columns={'CCI_14_0.015': 'CCI_14_0.015'})
    
    df_input = df[COLUNAS_INDICADORES].iloc[-1].fillna(0) 
    X_pred = df_input.values.reshape(1, -1)
    return X_pred, df_input 


def log_trade_data(timestamp_log, indicadores_df, probability, sinal):
    """Salva os dados brutos da predição no Supabase."""
    if supabase is None:
        print("Aviso: Supabase não inicializado. Não foi possível logar.")
        return

    # Mapeia os dados para as colunas da tabela 'wdo_trade_logs'
    # ATENÇÃO: Os nomes das chaves (keys) precisam bater exatamente com as colunas da sua tabela!
    data_to_insert = {
        "timestamp": timestamp_log,
        "probability": probability,
        "signal": sinal,
        "rsi_14": float(indicadores_df['RSI_14']), 
        "macdh_12_26_9": float(indicadores_df['MACDh_12_26_9']),
        "adx_14": float(indicadores_df['ADX_14']),
        "cci_14_0_015": float(indicadores_df['CCI_14_0.015']), # Mapeando para o nome de coluna (sem ponto)
        "bbp_5_2_0": float(indicadores_df['BBP_5_2.0']),
        "bbl_5_2_0": float(indicadores_df['BBL_5_2.0']),
        "atr_14": float(indicadores_df['ATR_14'])
    }

    try:
        # Usando insert().execute() para executar a query
        supabase.table(SUPABASE_TABLE).insert(data_to_insert).execute()
        print(f"Log de trade inserido: {sinal} ({probability:.4f})")
    except Exception as e:
        print(f"ERRO ao logar no Supabase: {e}")


@app.route('/', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde (Health Check)."""
    return jsonify({
        "status": "OK", 
        "message": "API de sinais WDO online."
    }), 200


@app.route('/signal', methods=['POST'])
def get_signal():
    """Endpoint principal para receber os dados e retornar o sinal."""
    if model_ia is None:
        return jsonify({"sinal": "ERRO", "mensagem": "Modelo não carregado na API"}), 500
        
    try:
        data = request.json
        historico = pd.DataFrame(data['candles'])
        historico = historico.rename(columns={'volume': 'tick_volume'})
        
        # 1. Prepara o Input e Calcula Indicadores
        X_pred, indicadores_df = preparar_input_ia(historico)
        
        # 2. Predição
        probabilidade = model_ia.predict(X_pred, verbose=0)[0][0]
        
        sinal = "AGUARDAR"
        if probabilidade >= THRESHOLD_COMPRA:
            sinal = "COMPRAR"
            
        # 3. LOGA OS DADOS NO SUPABASE (Não bloqueia a resposta da API)
        timestamp_log = int(historico.iloc[-1]['time']) 
        log_trade_data(timestamp_log, indicadores_df, probabilidade, sinal)
        
        # 4. Retorna o Sinal
        return jsonify({
            "sinal": sinal,
            "probabilidade": round(float(probabilidade), 4)
        })
        
    except Exception as e:
        print(f"ERRO durante a predição: {e}")
        return jsonify({"sinal": "ERRO", "mensagem": str(e)}), 500

