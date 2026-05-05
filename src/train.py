"""
Phase 3 — E-Commerce Customer Churn
Standalone training script: reads cleaned CSV, trains XGBoost, saves artefacts.

Usage:
    python src/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    roc_auc_score, average_precision_score,
)
from xgboost import XGBClassifier, plot_importance
import shap

# ── Portable paths ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PROC = os.path.join(ROOT, '..', 'data', 'processed', 'churn_cleaned.csv')
FIG_DIR   = os.path.join(ROOT, '..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROC)
print(f'Loaded data: {df.shape}')

# ── Encode ─────────────────────────────────────────────────────────────────────
categorical_feature = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_feature, drop_first=False)
df_encoded = df_encoded.drop(columns=['CustomerID'], errors='ignore')

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model ──────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='logloss',
    reg_alpha=1.0,
    reg_lambda=0.0,
)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          sample_weight=sample_weights,
          verbose=False)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('\n=== Results ===')
print('Accuracy :', accuracy_score(y_test, y_pred))
print('ROC AUC  :', roc_auc_score(y_test, y_proba))
print('Avg Prec :', average_precision_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# ── Plots ──────────────────────────────────────────────────────────────────────
def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize='true', cmap='Blues', ax=ax)
ax.set_title('Normalized Confusion Matrix')
_save('confusion_matrix.png')

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
ax.set_title('ROC Curve'); _save('roc_curve.png')

fig, ax = plt.subplots(figsize=(7, 5))
PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
ax.set_title('Precision-Recall Curve'); _save('pr_curve.png')

fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(model, max_num_features=20, importance_type='gain', ax=ax)
ax.set_title('Top 20 Feature Importances (Gain)'); _save('feature_importance.png')

# ── SHAP ───────────────────────────────────────────────────────────────────────
X_train_shap = X_train.astype('float64')
X_test_shap  = X_test.astype('float64')

background  = shap.sample(X_train_shap, 200, random_state=42)
explainer   = shap.TreeExplainer(model, data=background, model_output='probability')
shap_values = explainer(X_test_shap)

shap.plots.beeswarm(shap_values, max_display=20, show=False); _save('shap_beeswarm.png')

highest_risk_pos = np.argmax(y_proba)
shap.plots.waterfall(shap_values[highest_risk_pos], max_display=15, show=False)
_save('shap_waterfall_highrisk.png')

print(f'\nAll figures saved to {FIG_DIR}')
print('Done ✓')
