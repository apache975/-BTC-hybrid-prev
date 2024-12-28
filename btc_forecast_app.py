import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Função para carregar os dados
def carregar_dados():
    data = pd.read_csv('xgboost_data.csv')
    data['ds'] = pd.to_datetime(data['timestamp'])
    data['y'] = data['target']
    return data

data = carregar_dados()

# Configurações iniciais da aplicação Streamlit
st.title('Previsão de Preços do BTC com Cenários Interativos')
st.sidebar.header('Configurações dos Cenários')

# Ajustes interativos para os cenários
btc_dominance_delta = st.sidebar.slider('Ajuste de BTC Dominance (%)', -0.2, 0.2, 0.0, 0.01)
halving_impact_delta = st.sidebar.slider('Ajuste de Halving Impact (%)', -0.2, 0.2, 0.0, 0.01)
market_sentiment_delta = st.sidebar.slider('Ajuste de Sentimento de Mercado', -0.2, 0.2, 0.0, 0.01)
hash_rate_multiplier = st.sidebar.slider('Multiplicador de Hash Rate', 0.5, 2.0, 1.0, 0.1)
volume_price_ratio_multiplier = st.sidebar.slider('Multiplicador de Volume/Preço', 0.5, 2.0, 1.0, 0.1)

# Seleção de pesos para o modelo híbrido
xgb_weight = st.sidebar.slider('Peso do XGBoost no Modelo Híbrido', 0.0, 1.0, 0.2, 0.05)
prophet_weight = 1 - xgb_weight

# Adicionar novos regressores (simulação de dados adicionais)
data['market_sentiment'] = np.random.uniform(0.4, 0.6, len(data))  # Sentimento de mercado
data['hash_rate'] = np.random.uniform(1, 10, len(data))  # Taxa de Hash (exemplo)
data['volume_price_ratio'] = data['volume'] / data['close']  # Relação volume/preço

# Configurar e treinar o modelo Prophet
model_prophet = Prophet()
model_prophet.add_regressor('btc_dominance')
model_prophet.add_regressor('halving_impact')
model_prophet.add_regressor('market_sentiment')
model_prophet.add_regressor('hash_rate')
model_prophet.add_regressor('volume_price_ratio')
model_prophet.fit(data[['ds', 'y', 'btc_dominance', 'halving_impact', 'market_sentiment', 'hash_rate', 'volume_price_ratio']])

# Criar cenários futuros para previsões
future = model_prophet.make_future_dataframe(periods=730, freq='D')  # Previsões de 2 anos

# Cenário ajustado pelo utilizador
future_adjusted = future.copy()
future_adjusted['btc_dominance'] = data['btc_dominance'].iloc[-1] + btc_dominance_delta
future_adjusted['halving_impact'] = data['halving_impact'].iloc[-1] + halving_impact_delta
future_adjusted['market_sentiment'] = data['market_sentiment'].iloc[-1] + market_sentiment_delta
future_adjusted['hash_rate'] = data['hash_rate'].iloc[-1] * hash_rate_multiplier
future_adjusted['volume_price_ratio'] = data['volume_price_ratio'].iloc[-1] * volume_price_ratio_multiplier

# Fazer previsões com o Prophet
forecast_adjusted = model_prophet.predict(future_adjusted)

# Preparar os dados para o XGBoost
X = data.drop(['timestamp', 'target', 'ds', 'y'], axis=1)
y = data['target']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo XGBoost
model_xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)
model_xgb.fit(X_train, y_train)

# Fazer previsões do XGBoost para os dados históricos disponíveis
xgb_adjustments = model_xgb.predict(X.tail(len(data)))
xgb_adjustments_future = np.zeros(len(forecast_adjusted) - len(xgb_adjustments))  # Zeros para horizonte futuro

# Combinar previsões do histórico com zeros para o futuro
hybrid_forecast_adjusted = forecast_adjusted[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
hybrid_forecast_adjusted['xgb_adjustment'] = np.concatenate([xgb_adjustments, xgb_adjustments_future])
hybrid_forecast_adjusted['hybrid'] = (
    prophet_weight * hybrid_forecast_adjusted['yhat'] +
    xgb_weight * hybrid_forecast_adjusted['xgb_adjustment']
)

# Adicionar os dados reais ao gráfico
real_data = data[['ds', 'y']].rename(columns={'y': 'Dados Reais'})
hybrid_forecast_adjusted = hybrid_forecast_adjusted.merge(real_data, on='ds', how='left')

# Plotar o gráfico interativo com intervalos de confiança
st.subheader('Cenário Ajustado pelo Utilizador')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hybrid_forecast_adjusted['ds'], hybrid_forecast_adjusted['Dados Reais'], label='Dados Reais', color='blue')
ax.plot(hybrid_forecast_adjusted['ds'], hybrid_forecast_adjusted['hybrid'], label='Híbrido (Ajustado)', color='red')
ax.fill_between(hybrid_forecast_adjusted['ds'],
                hybrid_forecast_adjusted['yhat_lower'],
                hybrid_forecast_adjusted['yhat_upper'],
                color='gray', alpha=0.3, label='Intervalo de Confiança (Prophet)')
ax.legend()
ax.set_title('Previsão de Preços do BTC com Intervalos de Confiança')
ax.set_xlabel('Data')
ax.set_ylabel('Preço do BTC')
st.pyplot(fig)

# Avaliar o modelo XGBoost no conjunto de teste
y_pred_xgb = model_xgb.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=True) ** 0.5
#rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
st.sidebar.write(f"RMSE do XGBoost no conjunto de teste: {rmse_xgb}")
