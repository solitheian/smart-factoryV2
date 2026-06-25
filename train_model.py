import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import Deep Learning (LSTM) dari TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Loading data...")
# Load dataset
df = pd.read_csv('ai4i2020.csv')

# Drop kolom yang ga dipake
kolom_dibuang = ['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(kolom_dibuang + ['Machine failure'], axis=1, errors='ignore')
y = df['Machine failure']

# Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Mulai Adu Mekanik: Random Forest vs LSTM ---")

# ==========================================
# 1. TRAINING RANDOM FOREST
# ==========================================
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("Hasil Random Forest:")
y_pred_rf = rf_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# 2. TRAINING LSTM (DEEP LEARNING)
# ==========================================
print("\nTraining LSTM...")
# Reshape data buat LSTM (samples, time_steps, features)
# Karena tabular, time_steps kita set 1
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Bangun arsitektur LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
model_lstm.add(Dense(1, activation='sigmoid')) 

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM (epoch 10, batch_size 32)
model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_split=0.2)

print("\nHasil LSTM:")
y_pred_lstm_prob = model_lstm.predict(X_test_lstm)
# Convert probabilitas ke biner (0 atau 1)
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lstm):.3f}")
print(classification_report(y_test, y_pred_lstm))

# ==========================================
# 3. SAVING MODEL BUAT DEPLOYMENT
# ==========================================
print("\nNge-save model Random Forest, LSTM, dan Scaler buat Web App...")
joblib.dump(rf_model, 'model_mesin_rf.pkl')
model_lstm.save('model_mesin_lstm.h5') # Save model LSTM
joblib.dump(scaler, 'scaler_mesin.pkl')
print("✅ Model berhasil di-save! Siap buat di-deploy ke Streamlit.")