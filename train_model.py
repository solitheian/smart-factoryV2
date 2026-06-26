from sklearn.svm import SVC

# ==========================================
# 2. TRAINING SUPPORT VECTOR MACHINE (SVM)
# ==========================================
print("\nTraining Support Vector Machine (SVM)...")
# Kita pake probability=True biar nanti bisa dipake buat Streamlit/API
model_svm = SVC(kernel='rbf', probability=True, random_state=42)
model_svm.fit(X_train_scaled, y_train)

print("\nHasil SVM:")
y_pred_svm = model_svm.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")
print(classification_report(y_test, y_pred_svm))

# ==========================================
# 4. GENERATE VISUALISASI BARU (RF vs SVM)
# ==========================================
print("\n--- Generating Grafik Visualisasi... ---")

# Ambil metrik spesifik buat Class 1 (Risiko Kerusakan)
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, labels=[1], zero_division=0)
svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, labels=[1], zero_division=0)

rf_scores = [accuracy_score(y_test, y_pred_rf), rf_prec[0], rf_rec[0], rf_f1[0]]
svm_scores = [accuracy_score(y_test, y_pred_svm), svm_prec[0], svm_rec[0], svm_f1[0]]
metrics_labels = ['Accuracy', 'Precision\n(Class 1)', 'Recall\n(Class 1)', 'F1-Score\n(Class 1)']

# BIKIN BAR CHART
x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest (Tuned)', color='#2ecc71')
rects2 = ax.bar(x + width/2, svm_scores, width, label='SVM', color='#3498db')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Komparasi Metrik Performa: RF vs SVM', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.legend()
ax.set_ylim(0, 1.1)

# Tambahin angka di atas bar
ax.bar_label(rects1, fmt='%.2f', padding=3)
ax.bar_label(rects2, fmt='%.2f', padding=3)
plt.tight_layout()
plt.show()

# BIKIN CONFUSION MATRIX
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CM Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0], annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix - Random Forest', fontsize=12)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# CM SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 14})
axes[1].set_title('Confusion Matrix - SVM', fontsize=12)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# ==========================================
# 3. SAVING MODEL BUAT DEPLOYMENT
# ==========================================
print("\nNge-save model Random Forest dan SVM...")
joblib.dump(rf_model, 'model_mesin_rf.pkl')
joblib.dump(model_svm, 'model_mesin_svm.pkl')
joblib.dump(scaler, 'scaler_mesin.pkl')
print("✅ Revisi beres!")
