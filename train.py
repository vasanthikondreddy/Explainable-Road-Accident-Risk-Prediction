import pandas as pd, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

d = pd.read_csv("accidents_clean.csv")

cat = ["Light_Conditions_label", "Weather_Conditions_label", "Road_Surface_Conditions_label",
       "Road_Type_label", "Junction_Detail_label", "1st_Road_Class_label", "Urban_or_Rural_Area_label"]
num = ["hour", "Day_of_Week", "Speed_limit", "is_weekend"]

# Target: 1 = Fatal or Serious, 0 = Slight
y = d["Accident_Severity"].isin([1, 2]).astype(int)
X = d[cat + num]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
model = Pipeline([("pre", pre),
                  ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                        class_weight="balanced_subsample", n_jobs=-1, random_state=42))])
model.fit(Xtr, ytr)

p = model.predict_proba(Xte)[:, 1]
print("ROC-AUC:", round(roc_auc_score(yte, p), 3))
print("base rate (serious/fatal):", round(y.mean(), 3))

# pick a threshold that catches ~60% of serious/fatal accidents
for t in (0.3, 0.4, 0.5):
    pred = (p >= t).astype(int)
    print(f"\nthreshold {t}")
    print(classification_report(yte, pred, target_names=["Slight", "Serious/Fatal"], digits=3))

joblib.dump(model, "risk_model.pkl")
json.dump({"cat": cat, "num": num}, open("risk_features.json", "w"))

# top factors
names = model.named_steps["pre"].get_feature_names_out()
imp = pd.Series(model.named_steps["m"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop factors:\n", imp.head(10).round(3))