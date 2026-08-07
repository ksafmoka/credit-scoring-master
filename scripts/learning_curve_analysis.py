#!/usr/bin/env python3
"""
Learning Curve Analysis — diagnose underfitting vs overfitting

Run: docker compose exec airflow-scheduler python scripts/learning_curve_analysis.py
"""

import sys
from pathlib import Path

# Handle both Docker and local execution
docker_path = Path('/opt/airflow')
if docker_path.exists():
    sys.path.insert(0, str(docker_path))
else:
    ROOT = Path.cwd()
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from src.config import get_db_engine, TrainingConfig, ARTIFACTS_DIR
from src.data.queries import get_feature_dataset
from src.models.scoring.train import get_feature_matrix, get_model
from src.models.scoring.artifacts import ScoringArtifact

print('📊 Learning Curve Analysis')
print('='*70)

# Load data
engine = get_db_engine()
train, val, test = get_feature_dataset(engine, TrainingConfig.TRAIN_END_DATE, TrainingConfig.VAL_END_DATE)

print(f'\n✅ Data loaded:')
print(f'   Train: {len(train)} rows, default rate: {train[TrainingConfig.TARGET_COL].mean():.3%}')
print(f'   Val: {len(val)} rows, default rate: {val[TrainingConfig.TARGET_COL].mean():.3%}')
print(f'   Test: {len(test)} rows, default rate: {test[TrainingConfig.TARGET_COL].mean():.3%}')

# Load model
model_path = ARTIFACTS_DIR / 'with_history'
if (model_path / 'model.pkl').exists():
    artifact = ScoringArtifact.load(model_path)
    print(f'\n✅ Loaded model: {artifact.model_type}')
else:
    print(f'\n❌ No model found at {model_path}')
    sys.exit(1)

# Prepare features
X_train, feature_cols = get_feature_matrix(train, artifact.feature_names)
X_val, _ = get_feature_matrix(val, artifact.feature_names)

medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)

y_train = train[TrainingConfig.TARGET_COL].astype(int)
y_val = val[TrainingConfig.TARGET_COL].astype(int)

print(f'\n📈 Features: {len(feature_cols)}')
print(f'   Top 5: {feature_cols[:5]}')

# Train fresh model for learning curve
print(f'\n🔧 Training model for learning curve...')
model = get_model(artifact.model_type, {'n_estimators': 200, 'learning_rate': 0.05})
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score, precision_score, recall_score

train_pred = model.predict_proba(X_train)[:, 1]
val_pred = model.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)

print(f'\n📊 Current Performance:')
print(f'   Train AUC: {train_auc:.4f}')
print(f'   Val AUC: {val_auc:.4f}')
print(f'   Gap: {train_auc - val_auc:.4f}')

# Diagnose
if train_auc > 0.95:
    print(f'\n⚠️  OVERFITTING DETECTED!')
    print(f'   Train AUC is too high ({train_auc:.4f})')
    print(f'   Model memorized training data')
    print(f'   Solutions:')
    print(f'   - Add regularization')
    print(f'   - Reduce model complexity')
    print(f'   - Add more training data')
elif val_auc < 0.65:
    print(f'\n⚠️  UNDERFITTING DETECTED!')
    print(f'   Val AUC is too low ({val_auc:.4f})')
    print(f'   Model is too simple')
    print(f'   Solutions:')
    print(f'   - Add more features')
    print(f'   - Increase model complexity')
    print(f'   - Better feature engineering')
else:
    print(f'\n✅ Model looks balanced')

# Learning curve with different training sizes
print(f'\n📈 Computing learning curve...')

train_sizes = [1000, 5000, 10000, 20000, 50000, len(X_train)]
train_scores = []
val_scores = []

for size in train_sizes:
    if size > len(X_train):
        size = len(X_train)
    
    # Subsample
    X_sub = X_train.iloc[:size]
    y_sub = y_train.iloc[:size]
    
    # Train
    model_sub = get_model(artifact.model_type, {'n_estimators': 200, 'learning_rate': 0.05})
    model_sub.fit(X_sub, y_sub)
    
    # Evaluate
    train_pred = model_sub.predict_proba(X_sub)[:, 1]
    val_pred = model_sub.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_sub, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    
    train_scores.append(train_auc)
    val_scores.append(val_auc)
    
    print(f'   n={size:6d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Gap: {train_auc-val_auc:.4f}')

# Plot learning curve
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores, 'o-', label='Train AUC', color='blue')
plt.plot(train_sizes, val_scores, 'o-', label='Val AUC', color='red')
plt.xlabel('Training Set Size')
plt.ylabel('AUC')
plt.title('Learning Curve — Diagnose Under/Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig(ARTIFACTS_DIR / 'learning_curve.png', dpi=150)
print(f'\n✅ Learning curve saved to: {ARTIFACTS_DIR / "learning_curve.png"}')
plt.show()

# Interpretation
print(f'\n💡 Interpretation:')
if val_scores[-1] - val_scores[0] < 0.05:
    print(f'   - Val AUC barely improves with more data → model capacity issue')
    print(f'   - Need better features or more complex model')
elif train_scores[-1] - val_scores[-1] > 0.10:
    print(f'   - Large gap between train and val → overfitting')
    print(f'   - Need regularization or more data')
else:
    print(f'   - Learning curve looks healthy')
    print(f'   - Model is learning from data')

print(f'\n🎯 Recommendations:')
print(f'   1. Check if Val AUC improves with more data (learning curve)')
print(f'   2. If underfitting → add features (grade, sub_grade from LC)')
print(f'   3. If overfitting → add regularization, reduce complexity')
print(f'   4. Improve synthetic payment history (stronger correlation)')
print(f'   5. Consider ensemble of multiple models')

print(f'\n' + '='*70)
print(f'✅ Learning curve analysis complete!')
