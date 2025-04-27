import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_columns = joblib.load('feature_columns.pkl')

st.title('Product Review Score Prediction')

st.header('Enter product details')

price = st.number_input('Price of the product', min_value=0.0, step=0.1)
product_weight = st.number_input('Product weight (in grams)', min_value=0.0, step=0.1)

city_options = [
    "sao paulo", "rio de janeiro", "belo horizonte", "brasília", "curitiba", 
    "campinas", "porto alegre", "salvador", "guarulhos", "sao bernardo do campo", 
    "niteroi", "santo andre", "osasco", "goiania", "santos", "sao jose dos campos", 
    "fortaleza", "sorocaba", "recife", "florianopolis", "jundiai", "ribeirao preto", 
    "nova iguaçu", "belem", "contagem", "barueri", "juiz de fora", "sao goncalo", 
    "mogi das cruzes", "vitoria", "piracicaba", "uberlandia", "sao luis", 
    "sao jose do rio preto", "carapicuiba", "vila velha", "campo grande", "praia grande", 
    "maua", "londrina", "taboao da serra", "diadema", "indaiatuba", "maringa", 
    "serra", "taubate", "sao caetano do sul", "teresina", "duque de caxias", 
    "bauru", "joao pessoa", "cuiaba", "joinville", "petropolis", "sao carlos", 
    "cotia", "macae", "americana", "guaruja", "maceio", "suzano", "campos dos goytacazes", 
    "caxias do sul", "volta redonda"]
race_options = ['Black', 'White', 'Mixed / Other']  # Replace with your actual race options

city = st.selectbox('Select customer city', city_options)
race = st.selectbox('Select race', race_options)

if st.button('Predict Review Score'):
    input_data = {
        'price': price,
        'product_weight_g': product_weight,
        'customer_city_' + city: 1,
        'race_' + race: 1
    }

    for column in selected_columns:
        if column not in input_data:
            input_data[column] = 0

    input_df = pd.DataFrame([input_data])
    input_df[['price_scaled', 'product_weight_scaled']] = scaler.transform(
        input_df[['price', 'product_weight_g']]
    )

    prediction = model.predict(input_df[selected_columns])
    st.write(f'Predicted Review Score: {prediction[0]:.2f}')

