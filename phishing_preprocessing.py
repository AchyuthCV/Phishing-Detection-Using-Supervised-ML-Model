# ============================================================
#  Phishing URL Dataset — Data Preprocessing Pipeline
#  Picks up exactly where the Kaggle notebook left off
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
# Update this path to wherever your dataset.csv is stored
df = pd.read_csv('dataset.csv')

print("=" * 55)
print("STEP 1 — Raw Data Loaded")
print("=" * 55)
print(f"  Shape        : {df.shape}")
print(f"  Columns      : {df.shape[1]}")
print(f"  Missing vals : {df.isnull().sum().sum()}")


# ── 2. DROP UNNECESSARY COLUMNS ──────────────────────────────
# 'index' is just a row ID — no predictive value
df = df.drop(columns=['index'])

print("\n" + "=" * 55)
print("STEP 2 — Dropped 'index' column")
print(f"  New shape    : {df.shape}")


# ── 3. FIX LABEL ENCODING ────────────────────────────────────
# Original labels: 1 = Legitimate, -1 = Phishing
# We remap -1 → 0 so it becomes a clean binary problem:
#   0 = Phishing   |   1 = Legitimate

df['Result'] = df['Result'].replace(-1, 0)

print("\n" + "=" * 55)
print("STEP 3 — Remapped Target Labels  (-1 → 0)")
print(f"  Class distribution:\n{df['Result'].value_counts().to_string()}")
print(f"  Phishing  : {(df['Result']==0).sum()} ({(df['Result']==0).mean()*100:.1f}%)")
print(f"  Legitimate: {(df['Result']==1).sum()} ({(df['Result']==1).mean()*100:.1f}%)")


# ── 4. REMAP FEATURE VALUES (0 stays, -1 → 0 for features) ──
# Features use:  1=legitimate, 0=suspicious, -1=phishing
# Keeping -1 can confuse tree models (ordinal assumption).
# We remap feature -1 → 0, making all features binary/ternary
# with values in {0, 1} or {0, 0.5, 1}.

# Strategy: map {-1: 0, 0: 0.5, 1: 1} for ternary features
#           map {-1: 0, 1: 1}          for binary features

feature_cols = [c for c in df.columns if c != 'Result']

binary_features  = []   # only 2 unique values
ternary_features = []   # 3 unique values (includes 0)

for col in feature_cols:
    unique = set(df[col].unique())
    if unique == {-1, 1}:
        binary_features.append(col)
    else:
        ternary_features.append(col)

# Remap binary features: -1 → 0
for col in binary_features:
    df[col] = df[col].replace(-1, 0)

# Remap ternary features: -1 → 0, 0 → 0.5, 1 → 1
for col in ternary_features:
    df[col] = df[col].replace({-1: 0, 0: 0.5, 1: 1})

print("\n" + "=" * 55)
print("STEP 4 — Feature Value Remapping")
print(f"  Binary features  ({len(binary_features)}) : -1 → 0, 1 → 1")
print(f"  Ternary features ({len(ternary_features)}) : -1 → 0, 0 → 0.5, 1 → 1")
print(f"  Sample binary   : {binary_features[:4]}")
print(f"  Sample ternary  : {ternary_features[:4]}")


# ── 5. DROP HIGHLY CORRELATED FEATURES ───────────────────────
# The notebook noted popUpWindow & Favicon are highly correlated.
# We compute full correlation matrix and drop one of each pair
# with correlation > 0.90.

corr_matrix = df[feature_cols].corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = [
    (col, row, upper_tri.loc[row, col])
    for col in upper_tri.columns
    for row in upper_tri.index
    if pd.notna(upper_tri.loc[row, col]) and upper_tri.loc[row, col] > 0.90
]

cols_to_drop_corr = list(set([pair[1] for pair in high_corr_pairs]))
df = df.drop(columns=cols_to_drop_corr)
feature_cols = [c for c in df.columns if c != 'Result']

print("\n" + "=" * 55)
print("STEP 5 — Removed Highly Correlated Features (threshold > 0.90)")
if high_corr_pairs:
    for a, b, v in high_corr_pairs:
        print(f"  Dropped '{b}' (corr with '{a}' = {v:.2f})")
else:
    print("  No feature pairs exceeded the 0.90 threshold.")
print(f"  Remaining features: {len(feature_cols)}")


# ── 6. FEATURE SELECTION (Chi-2) ─────────────────────────────
# Select top K features using chi-squared test.
# Since all values are now in [0, 1], chi2 is valid here.

X = df[feature_cols]
y = df['Result']

K = min(20, len(feature_cols))   # keep top 20 or all if fewer
selector = SelectKBest(chi2, k=K)
selector.fit(X, y)

selected_mask   = selector.get_support()
selected_feats  = X.columns[selected_mask].tolist()
dropped_feats   = X.columns[~selected_mask].tolist()

X_selected = pd.DataFrame(selector.transform(X), columns=selected_feats)

print("\n" + "=" * 55)
print(f"STEP 6 — Chi-2 Feature Selection (top {K})")
print(f"  Selected : {selected_feats}")
if dropped_feats:
    print(f"  Dropped  : {dropped_feats}")


# ── 7. TRAIN / TEST SPLIT ────────────────────────────────────
# 80% train, 20% test — stratified to preserve class balance

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 55)
print("STEP 7 — Train / Test Split  (80 / 20, stratified)")
print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")
print(f"  Train class balance : {y_train.value_counts().to_dict()}")
print(f"  Test  class balance : {y_test.value_counts().to_dict()}")


# ── 8. SCALING (optional for tree models, useful for others) ──
# XGBoost doesn't need scaling, but Logistic Regression / SVM do.
# We prepare both scaled and unscaled versions.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)         # transform test

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_feats)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=selected_feats)

print("\n" + "=" * 55)
print("STEP 8 — Standard Scaling Applied")
print("  (Fit on train only — no data leakage)")
print(f"  X_train_scaled mean (first 3): "
      f"{X_train_scaled.iloc[:, :3].mean().round(4).values}")
print(f"  X_train_scaled std  (first 3): "
      f"{X_train_scaled.iloc[:, :3].std().round(4).values}")


# ── 9. SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("✅  PREPROCESSING COMPLETE — Data Ready for Modelling")
print("=" * 55)
print(f"""
  Outputs available:
  ┌─────────────────────────────────────────────┐
  │  For tree models (XGBoost, RandomForest):   │
  │    X_train, X_test, y_train, y_test         │
  │                                             │
  │  For linear models (LogReg, SVM, KNN):      │
  │    X_train_scaled, X_test_scaled            │
  │    y_train, y_test                          │
  │                                             │
  │  Feature count : {len(selected_feats):<3}                          │
  │  Train samples : {len(X_train):<5}                        │
  │  Test  samples : {len(X_test):<5}                        │
  └─────────────────────────────────────────────┘
""")
