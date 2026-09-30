import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch 
import torch.nn as nn


class RidgeRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RidgeRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Configurazione della pagina
st.set_page_config(page_title="Predittore Valore Calciatori", layout="centered")
st.title("⚽ Stima del Valore di Mercato")
st.write("Inserisci le statistiche del giocatore e scegli l'algoritmo per ottenere una stima.")

# Caricamento dei modelli
@st.cache_resource
def load_models():
    # Caricamento Random Forest
    rf = joblib.load('random_forest_model.pkl')
    
    # Carica Ridge Regression
    # Inizializzazione del modello vuoto
    ridge = RidgeRegressionModel(input_dim=44) 
    
    # Caricamento dei pesi salvati
    ridge.load_state_dict(torch.load('ridge_pesi.pth', map_location=torch.device('cpu')))
    ridge.eval()
    
    return rf, ridge

rf_model, ridge_model = load_models()

# Barra laterale per la scelta del modello
st.sidebar.header("Impostazioni Predizione")
modello_scelto = st.sidebar.radio(
    "Scegli l'algoritmo predittivo:",
    ["Random Forest", "Ridge Regression (PyTorch)"]
)

# Interfaccia di Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Anagrafica e Impiego")
    age = st.number_input("Età", min_value=15, max_value=55, value=27)
    
    appearances = st.number_input("Presenze", min_value=0, max_value=2000, value=180)
    minutes = st.number_input("Minuti Giocati", min_value=0, max_value=200000, value=14500)
    
    ruolo = st.selectbox("Ruolo", ["Attack", "Midfield", "Defender", "Goalkeeper"])

with col2:
    st.subheader("Metriche Offensive e Lega")
    
    goals = st.number_input("Gol", min_value=0, max_value=2000, value=5)
    assists = st.number_input("Assist", min_value=0, max_value=2000, value=18)
    
    lega = st.selectbox("Campionato", ["Premier League", "Serie A", "La Liga", "Bundesliga", "Ligue 1"])

# Calcolo e Predizione
if st.button(f"Calcola con {modello_scelto}", type="primary"):
    
    # 1. Calcolo automatico delle feature derivate
    goals_per_90 = (goals / minutes) * 90 if minutes > 0 else 0
    assists_per_90 = (assists / minutes) * 90 if minutes > 0 else 0
    has_appearance_data = 1 if appearances > 0 else 0

    # 2. COSTRUZIONE DELLE FEATURE NUMERICHE (1 riga, 7 colonne)
    numeric_array = np.array([[
        age, appearances, minutes, goals, assists, goals_per_90, assists_per_90
    ]])

    # 3. COSTRUZIONE DELLE FEATURE CATEGORICHE (1 riga, 9 colonne)
    cat_array = np.zeros((1, 9)) 
    
    cat_array[0, 0] = has_appearance_data

    # Ruoli (Baseline: "Attack" e "Missing" = 0)
    if ruolo == "Defender":     cat_array[0, 1] = 1
    elif ruolo == "Goalkeeper": cat_array[0, 2] = 1
    elif ruolo == "Midfield":   cat_array[0, 3] = 1

    # Campionati (Baseline: "La Liga" = 0)
    if lega == "Ligue 1":          cat_array[0, 5] = 1
    elif lega == "Premier League": cat_array[0, 6] = 1
    elif lega == "Serie A":        cat_array[0, 7] = 1
    elif lega == "Bundesliga":     cat_array[0, 8] = 1

    # 4. UNIONE E STANDARDIZZAZIONE (sulle 16 feature originali)
    base_features = np.hstack([numeric_array, cat_array]) # Shape (1, 16)
    
    mu = np.load('mu_poly.npy')
    sigma = np.load('sigma_poly.npy')
    sigma = np.where(sigma == 0, 1e-8, sigma)
    
    base_features_norm = (base_features - mu) / sigma

    # 5. SEPARAZIONE E POLINOMIO (solo sulle 7 numeriche normalizzate)
    numeric_norm = base_features_norm[:, :7]
    cat_norm = base_features_norm[:, 7:]
    
    poly = joblib.load('poly_features.pkl')
    numeric_poly = poly.transform(numeric_norm) # Diventano 35

    # 6. AGGREGAZIONE FINALE (35 + 9 = 44 feature)
    X_input = np.hstack([numeric_poly, cat_norm])

    # 7. INFERENZA SUI MODELLI REALI
    if modello_scelto == "Random Forest":
        pred_log = rf_model.predict(X_input)
        valore_milioni = np.expm1(pred_log)[0]
        
    elif modello_scelto == "Ridge Regression (PyTorch)":
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad(): 
            pred_log = ridge_model(X_tensor)
        valore_milioni = np.expm1(pred_log.numpy())[0][0]

    # 8. RISULTATO A SCHERMO
    st.success(f"💶 Valore Stimato ({modello_scelto}): **{valore_milioni:.1f} Milioni di Euro**")
    st.balloons()