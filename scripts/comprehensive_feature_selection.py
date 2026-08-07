#!/usr/bin/env python3
"""
Comprehensive Feature Selection from Lending Club (151 features)
Run: python scripts/comprehensive_feature_selection.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('COMPREHENSIVE FEATURE SELECTION FROM LENDING CLUB')
print('='*80)

# 1. Load data
print('\n📊 Loading Lending Club data...')
df = pd.read_csv('data/lending_club.csv', low_memory=False)
print(f'✅ Loaded: {len(df):,} rows × {len(df.columns)} columns')

# 2. Exclude leakage and irrelevant features
exclude = [
    'id', 'member_id',  # IDs
    'url', 'desc', 'title', 'emp_title',  # Text/URLs
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',  # Leakage
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
    'loan_status'  # Target
]

candidate_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
print(f'\n🔍 Candidate features: {len(candidate_cols)} (excluded {len(exclude)} leakage)')

# 3. Prepare target
default_statuses = {'Charged Off', 'Default', 'Late (31-120 days)'}
df['is_default'] = df['loan_status'].isin(default_statuses).astype(int)
print(f'🎯 Target: is_default (default rate: {df["is_default"].mean():.2%})')

# 4. Prepare model data
df_model = df[candidate_cols + ['is_default']].copy()
df_model = df_model.fillna(df_model.median())
print(f'✅ Model ready: {len(df_model):,} rows × {len(candidate_cols)} features')

# 5. Train model and compute importance
print(f'\n🔧 Training model (sampling 50k for speed)...')
X = df_model.drop('is_default', axis=1)
y = df_model['is_default']

sample_idx = np.random.choice(len(X), min(50000, len(X)), replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_sample, y_sample)

print(f'🔍 Computing permutation importance (PR-AUC)...')
result = permutation_importance(
    model, X_sample, y_sample,
    n_repeats=10, random_state=42,
    scoring='average_precision'
)

perm_importance = pd.DataFrame({
    'feature': candidate_cols,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

# 6. Remove redundant features (correlation > 0.85)
print(f'\n🔍 Computing correlation matrix...')
corr = X_sample.corr()
redundant = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.85:
            f1, f2 = corr.columns[i], corr.columns[j]
            imp1 = perm_importance[perm_importance['feature'] == f1]['importance'].values[0]
            imp2 = perm_importance[perm_importance['feature'] == f2]['importance'].values[0]
            redundant.add(f1 if imp1 < imp2 else f2)

print(f'📊 Redundant features removed: {len(redundant)}')

# 7. Select final pool
threshold = perm_importance['importance'].median() * 0.1
final_features = perm_importance[
    (perm_importance['importance'] >= threshold) &
    (~perm_importance['feature'].isin(redundant))
]['feature'].tolist()

print(f'\n🎯 FINAL FEATURE POOL: {len(final_features)} features')

# 8. Save results
import os
os.makedirs('artifacts', exist_ok=True)

perm_importance.to_csv('artifacts/feature_importance_all.csv', index=False)
with open('artifacts/final_feature_pool.txt', 'w') as f:
    f.write('\n'.join(final_features))

print(f'\n✅ Saved:')
print(f'   - artifacts/feature_importance_all.csv (all features)')
print(f'   - artifacts/final_feature_pool.txt (final pool)')

print(f'\n📊 Top 30 Features:')
print(perm_importance.head(30).to_string(index=False))

print(f'\n📝 Final Feature Pool ({len(final_features)} features):')
for i, feat in enumerate(final_features, 1):
    imp = perm_importance[perm_importance['feature'] == feat]['importance'].values[0]
    print(f'   {i:2d}. {feat:<40} ({imp:.6f})')

print('\n' + '='*80)
print('✅ Comprehensive feature selection complete!')
print('='*80)
