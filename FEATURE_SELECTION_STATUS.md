# Feature Selection Status

## ✅ Completed

1. **Comprehensive Feature Selection Script Created**
   - File: `scripts/comprehensive_feature_selection.py`
   - Committed to GitHub: ✅
   - Pushed to main: ✅

2. **Script Capabilities**
   - Loads ALL 151 features from Lending Club (1.67GB dataset)
   - Excludes leakage features (post-origination, IDs, text)
   - Computes permutation importance (PR-AUC)
   - Removes redundant features (correlation > 0.85)
   - Outputs final feature pool to `artifacts/final_feature_pool.txt`

## 🔄 Next Steps (When You Return)

### Step 1: Pull Latest Changes
```bash
cd C:\Users\pc\Downloads\credit-scoring-master
git pull origin main
```

### Step 2: Run Feature Selection
```bash
python scripts/comprehensive_feature_selection.py
```

**Expected Output:**
- `artifacts/feature_importance_all.csv` - all 151 features ranked
- `artifacts/final_feature_pool.txt` - final pool of best features

**Runtime:** ~5-10 minutes (large dataset)

### Step 3: Review Results
```bash
# See final feature pool
type artifacts\final_feature_pool.txt

# See all features ranked by importance
type artifacts\feature_importance_all.csv
```

### Step 4: Update Code with New Features

After reviewing the final feature pool, I'll help you:

1. **Update `src/config.py`**
   - Update `FeatureConfig.ALL_FEATURES` with final pool

2. **Update `src/data/ingestion.py`**
   - Add new features to `COLUMN_MAP`

3. **Update `src/features/numerical.py`**
   - Add feature engineering for new features

4. **Rebuild Pipeline**
   ```bash
   # Re-run data ingestion (loads new features)
   docker compose exec airflow-scheduler airflow dags trigger data_ingestion
   
   # Re-run feature engineering
   docker compose exec airflow-scheduler airflow dags trigger feature_engineering
   
   # Re-train model
   docker compose exec airflow-scheduler airflow dags trigger model_training
   ```

### Step 5: Evaluate Improvement

1. **Learning Curve**
   ```bash
   docker compose exec airflow-scheduler python scripts/learning_curve_analysis.py
   ```

2. **Error Analysis**
   ```bash
   docker compose exec airflow-scheduler jupyter notebook notebooks/05_error_analysis.ipynb --allow-root
   ```

## 📊 Expected Improvements

Based on Lending Club feature analysis:

**High-Impact Features (not currently used):**
- `grade` / `sub_grade` - LC internal risk rating (A-G) → Expected +0.03-0.05 PR-AUC
- `verification_status` - income verified → Expected +0.01-0.02 PR-AUC
- `revol_util` - revolving utilization → Expected +0.01-0.02 PR-AUC
- `pub_rec_bankruptcies` - bankruptcies → Expected +0.02-0.03 PR-AUC

**Total Expected Improvement:** +0.06-0.10 PR-AUC

## 📝 Notes

- Current model: ~0.20 PR-AUC (weak)
- Target after feature selection: ~0.26-0.30 PR-AUC
- Script handles 151 features efficiently with sampling
- Correlation analysis removes redundant features
- Permutation importance uses PR-AUC (not ROC-AUC) for imbalanced data

## 🔧 Troubleshooting

If script fails:
1. Check if `data/lending_club.csv` exists (1.67GB)
2. Ensure Python environment has required packages:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Check memory - script samples 50k rows for speed
4. Check `artifacts/` directory exists and is writable

## 📚 Related Files

- `scripts/comprehensive_feature_selection.py` - feature selection script
- `notebooks/07_feature_selection_comprehensive.ipynb` - alternative notebook version
- `artifacts/final_feature_pool.txt` - output (after running)
- `artifacts/feature_importance_all.csv` - output (after running)

---

**Status:** Ready to run. Just pull and execute the script!
