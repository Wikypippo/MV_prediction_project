# ⚽ Predizione del Valore di Mercato dei Calciatori

Questo progetto di Machine Learning esplora e prevede il valore di mercato dei calciatori militanti nei top 5 campionati europei. Utilizzando i dati estratti dal portale Transfermarkt, il progetto confronta le prestazioni e l'interpretabilità di un algoritmo ad albero (Random Forest) con un modello lineare regolarizzato implementato tramite PyTorch.

## 📊 Il Dataset e Preprocessing
I dati includono statistiche anagrafiche, di impiego sul campo (minuti giocati, presenze) e di efficienza offensiva (gol, assist, e relative metriche `per_90`), oltre a variabili categoriche per il ruolo tattico e la lega di appartenenza.

Per ottimizzare l'addestramento, la pipeline di preprocessing include:
* **Trasformazione Logaritmica:** Applicata alla variabile target (valore in Euro) per stabilizzare la varianza.
* **Polynomial Features (d=2):** Utilizzate esclusivamente per il modello lineare per simulare le non-linearità (es. curva di invecchiamento).
* **Standard Scaling:** Per standardizzare le feature numeriche ed evitare squilibri nell'aggiornamento dei pesi della rete neurale.
* **One-Hot Encoding:** Per la vettorizzazione dei ruoli e dei campionati.

## 🧠 Modelli Implementati
1. **Ridge Regression (PyTorch):** Modello lineare addestrato tramite discesa del gradiente. Implementa una penalizzazione L2 (`weight_decay`) per mitigare l'overfitting causato dall'espansione polinomiale (44 feature totali).
2. **Random Forest Regressor (Scikit-Learn):** Algoritmo di ensemble addestrato sulle 16 feature base, ideale per catturare regole decisionali complesse senza richiedere trasformazioni polinomiali.

## 🛠️ Tecnologie Utilizzate
* Python 3
* PyTorch (Deep Learning & Linear Modeling)
* Scikit-Learn (Random Forest, Preprocessing, Metriche)
* Pandas & NumPy (Data Manipulation)
* Matplotlib (Data Visualization)