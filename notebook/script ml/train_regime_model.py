import pandas as pd, lightgbm as lgb, pathlib, json
from datetime import date, timedelta

DATA_DIR = pathlib.Path("output/daily_snapshots")
snap_files = sorted(DATA_DIR.glob("snapshots_*.csv"))

df = pd.concat(pd.read_csv(f) for f in snap_files).sort_values("date")
df["date"] = pd.to_datetime(df["date"])

# ----------  create label  -----------
df["fwd_return"] = df["spot"].pct_change().shift(-1)      # spot(t+1)/spot(t) - 1
df["label"] = (df["fwd_return"] > 0.003).astype(int)      # bullish if > +0.3%

# drop last row (no forward return) & NaNs
df = df.dropna(subset=["label"]).reset_index(drop=True)

FEATURES = ["net_delta", "total_gamma", "total_vanna", "total_charm"]
X, y = df[FEATURES], df["label"]

train_frac = 0.8
cut = int(len(df)*train_frac)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1)
model.fit(X_train, y_train)

print("AUC on hold-out:",
      lgb.LGBMClassifier().fit(X_train, y_train).score(X_test, y_test))

# save model
import joblib, pathlib
MODEL_PATH = pathlib.Path("models")
MODEL_PATH.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH / "gef_regime_model.pkl")
print("✅ model saved.")
