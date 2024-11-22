import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, request, jsonify, Response
import psutil
import time
from prometheus_client import generate_latest, Gauge, Counter, Histogram, start_http_server, CONTENT_TYPE_LATEST, CollectorRegistry
from flask_restx import Api, Resource, fields



# Inicializar o Flask e Prometheus
app = Flask(__name__)
app.json.sort_keys = False
api = Api(app,
            version='1.0.0',
            title='API TECH 4 @mrvluiz',
            description='API para previsibilidade de preços de ações da bolsa',
            default='Principal'
            )



# métricas do Prometheus 
registry = CollectorRegistry()
REQUEST_COUNT = Counter('api_request_count', 'Total de requisições para a API', registry=registry)
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Latência das requisições', registry=registry)
CPU_USAGE = Gauge('cpu_usage_percent', 'Uso de CPU em porcentagem', registry=registry)
MEMORY_USAGE = Gauge('memory_usage_percent', 'Uso de memória em porcentagem', registry=registry)


# Função para monitorar CPU e memória
def monitor_resources():
    CPU_USAGE.set(psutil.cpu_percent(interval=0))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

# Função de pré-processamento dos dados
def preprocess_data(stock_data, n_steps=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

# Função para construir e treinar o modelo LSTM
def build_and_train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    
    return model

# Função para prever preços futuros
def prever_precos(model, scaler, data, n_steps=30, dias_futuros=120):
    data_scaled = scaler.transform(data)
    input_sequence = data_scaled[-n_steps:]
    input_sequence = input_sequence.reshape(1, n_steps, 1)
    
    predictions = []
    for _ in range(dias_futuros):
        prediction = model.predict(input_sequence)
        predictions.append(prediction[0][0])
        input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten().tolist()


# Função para carregar o modelo salvo
def load_or_train_model(X_train, y_train, X_val, y_val, model_path='stock_predict_model.h5'):
    if os.path.exists(model_path):
        print("Carregando o modelo salvo...")
        return load_model(model_path)
    else:
        print("Modelo não encontrado. Treinando um novo modelo...")
        model = build_and_train_model(X_train, y_train, X_val, y_val)
        model.save(model_path)        
        return model


def avaliar_model(model, X_test, y_test, scaler):    
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Desnormalizar os valores previstos e reais
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Cálculo das métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return mae, rmse, mape

def predicao(codigo_acao,data_inicial, data_final, dias_futuros_previsao ):
    
    REQUEST_COUNT.inc()  # Incrementar o contador de requisições
    start_time = time.time()

    # dados do JSON do Post da API
    data = request.get_json()
    codigo_acao = data.get('codigo_acao', 'MGLU3.SA')
    data_inicial = data.get('data_inicial', '2020-01-01')
    data_final = data.get('data_final', '2023-01-01')
    dias_futuros_previsao = data.get('dias_futuros_previsao', 120)

    # Obtem as informações da ação recebida no parametro
    yf_return = yf.download(codigo_acao, start=data_inicial, end=data_final)
    dados_historicos_acao = yf_return[['Close']]
       
    # Pré-processamento com os dados historicos
    X, y, scaler = preprocess_data(dados_historicos_acao)


    # Treinar ou carregar modelo caso ja tenha sido salvo
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    model = load_or_train_model(X_train, y_train, X_val, y_val)


    # Avalia o modelo
    mae, rmse, mape = avaliar_model(model, X_val, y_val, scaler)

    # Realiza a previsão dos preços futuros
    precos_futuros = prever_precos(model, scaler, dados_historicos_acao.values, dias_futuros=dias_futuros_previsao)
    

    #pega os dados históricos da mesma quantidade de dias informada no futuro  apenas para um melhor contexto.
    try:
        dias_historicos = min(dias_futuros_previsao, len(dados_historicos_acao))
        precos_historicos = dados_historicos_acao[-dias_historicos:].values.flatten().tolist()
    except:
        precos_historicos = dados_historicos_acao

    # Monitoraramento 
    monitor_resources()
    latency = time.time() - start_time

    return jsonify({
        'precos_historicos': precos_historicos, 
        'precos_futuros': precos_futuros,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'latencia': latency
    })




# Modelo de entrada (para documentação)
input_model = api.model('Predicao Model', {
    'codigo_acao': fields.String(required=True, description="Codigo da ação da Bolsa", default="MGLU3.SA"),
    'data_inicial': fields.String(required=True, description="Data Inicial para busca histórica", default="2020-01-01"),
    'data_final': fields.String(required=True, description="Data Final para busca histórica",  default="2024-01-01"),
    'dias_futuros_previsao': fields.Integer(required=True, description="Dias para previsão",  default=120)
})

@api.route("/predicao/", methods=['POST'])
@api.doc(description="Retorna a previsão de preços da ação informada")
class Preditiva(Resource):        
        @api.expect(input_model) 
        @REQUEST_LATENCY.time()  # Medir latência com Prometheus
        def post(self):                  
                # Obter os parâmetros do corpo da requisição
                data = request.get_json()
                codigo_acao = data.get('codigo_acao')
                data_inicial = data.get('data_inicial')
                data_final = data.get('data_final')
                dias_futuros_previsao = data.get('dias_futuros_previsao')

                # Verificar se os parâmetros foram enviados
                if not codigo_acao or not data_inicial or not data_final or not dias_futuros_previsao:
                    return {"error": "Os campos 'codigo_acao' , 'data_inicial', 'data_final' e 'dias_futuros_previsao' são obrigatórios."}, 400

                return predicao (codigo_acao,data_inicial, data_final, dias_futuros_previsao)
                        

@api.route("/metricas")
@api.doc(description="Retorna as metricas de monitoramento baseado no Prometeus (em GET)")
class Monitoramento(Resource):        
        def get(self):                  
                return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)
                # retornando conforme retorno do prometeus


if __name__ == '__main__':        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    # Iniciar o Prometheus em uma porta separada da aplicação Principal
    #start_http_server(8001) 
    # Removido por incompatibilidade com o Heroku.

    
