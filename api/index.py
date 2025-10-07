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
# O Vercel define estas variáveis de ambiente (Secrets)
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
    """Carrega o modelo e inicializa o cliente Supabase."""
    global model_ia, supabase
    
   # Carrega Modelo
    try:
        # CORREÇÃO: Usando compile=False para evitar erros de compatibilidade no Vercel
        model_ia = tf.keras.models.load_model('ia_wdo_v1_ticks.h5', compile=False) 
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        
    # Inicializa Supabase (Só conecta se as Secrets forem definidas)
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"ERRO ao inicializar Supabase: {e}")

inicializar_servicos()


def preparar_input_ia(historico: pd.DataFrame):
    """Calcula os 7 indicadores e retorna o array para previsão e os valores para log."""
    df = historico[['open', 'high', 'low', 'close', 'tick_volume']].copy()
    
    # Cálculo dos Indicadores
    df.ta.bbands(close='close', length=5, append=True) 
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.cci(length=14, append=True)

    df_input = df[COLUNAS_INDICADORES].iloc[-1].fillna(0) 
    X_pred = df_input.values.reshape(1, -1)
    return X_pred, df_input 


def log_trade_data(timestamp_log, indicadores_df, probability, sinal):
    """Salva os dados brutos da predição no Supabase."""
    if supabase is None:
        return

    # Mapeia os dados para as colunas da tabela 'wdo_trade_logs'
    data_to_insert = {
        "timestamp": timestamp_log,
        "probability": probability,
        "signal": sinal,
        "rsi_14": float(indicadores_df['RSI_14']), 
        "macdh_12_26_9": float(indicadores_df['MACDh_12_26_9']),
        "adx_14": float(indicadores_df['ADX_14']),
        "cci_14_0_015": float(indicadores_df['CCI_14_0.015']),
        "bbp_5_2_0": float(indicadores_df['BBP_5_2.0']),
        "bbl_5_2_0": float(indicadores_df['BBL_5_2.0']),
        "atr_14": float(indicadores_df['ATR_14'])
    }

    try:
        supabase.table(SUPABASE_TABLE).insert(data_to_insert).execute()
    except Exception as e:
        # Se houver um erro de conexão/API, o Vercel não travará, apenas não logará.
        pass


@app.route('/signal', methods=['POST'])
def get_signal():
    """Endpoint principal para receber os dados e retornar o sinal."""
    if model_ia is None:
        return jsonify({"sinal": "ERRO", "mensagem": "Modelo não carregado"}), 500
        
    try:
        data = request.json
        historico = pd.DataFrame(data['candles'])
        historico = historico.rename(columns={'volume': 'tick_volume'})
        
        X_pred, indicadores_df = preparar_input_ia(historico)
        
        probabilidade = model_ia.predict(X_pred, verbose=0)[0][0]
        
        sinal = "AGUARDAR"
        if probabilidade >= THRESHOLD_COMPRA:
            sinal = "COMPRAR"
            
        # LOGA OS DADOS NO SUPABASE
        timestamp_log = int(historico.iloc[-1]['time']) 
        log_trade_data(timestamp_log, indicadores_df, probabilidade, sinal)
        
        return jsonify({
            "sinal": sinal,
            "probabilidade": round(float(probabilidade), 4)
        })
    
    except Exception as e:
        return jsonify({"sinal": "ERRO", "mensagem": str(e)}), 500
