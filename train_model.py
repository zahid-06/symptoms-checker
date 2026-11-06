import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"

DISEASE_RENAME = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Osteoarthristis": "Osteoarthritis",
    "(vertigo) Paroymsal Positional Vertigo": "(Vertigo) Paroxysmal Positional Vertigo",
    "hepatitis A": "Hepatitis A",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "Dimorphic hemorrhoids(piles)": "Dimorphic hemorrhoids (piles)",
}


def clean_symptom(text):
    """' skin_rash' -> 'skin_rash', 'dischromic _patches' -> 'dischromic_patches'"""
    if not isinstance(text, str):
        return None
    return re.sub(r"_+", "_", text.strip().replace(" ", "_"))


def clean_disease(text):
    text = re.sub(r"\s+", " ", text.strip())
    return DISEASE_RENAME.get(text, text)



data = pd.read_csv(DATA / "dataset.csv")
severity_df = pd.read_csv(DATA / "Symptom-severity.csv")
desc_df = pd.read_csv(DATA / "symptom_Description.csv")
prec_df = pd.read_csv(DATA / "symptom_precaution.csv")
symptom_cols = [c for c in data.columns if c.startswith("Symptom")]

data["Disease"] = data["Disease"].map(clean_disease)
for col in symptom_cols:
    data[col] = data[col].map(clean_symptom)
severity_df["Symptom"] = severity_df["Symptom"].map(clean_symptom).replace(
    {"foul_smell_ofurine": "foul_smell_of_urine"}
)
severity_df = severity_df[severity_df["Symptom"] != "prognosis"]
desc_df["Disease"] = desc_df["Disease"].map(clean_disease)
prec_df["Disease"] = prec_df["Disease"].map(clean_disease)


all_symptoms = sorted({s for s in data[symptom_cols].values.ravel() if isinstance(s, str)})
X = pd.DataFrame(0, index=data.index, columns=all_symptoms)
for col in symptom_cols:
    for row, symptom in data[col].items():
        if isinstance(symptom, str):
            X.loc[row, symptom] = 1
y = data["Disease"]
print(f"Rows: {len(X)} | Diseases: {y.nunique()} | Symptoms: {len(all_symptoms)}")


combined = pd.concat([X, y], axis=1).drop_duplicates()
X, y = combined.drop(columns="Disease"), combined["Disease"]
print(f"Unique rows after removing duplicates: {len(X)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": BernoulliNB(),
}
print("\n--- Cross-validation accuracy ---")
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")


final_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)


artifacts = {
    "model": final_model,
    "symptoms": all_symptoms,
    "severity": dict(zip(severity_df["Symptom"], severity_df["weight"].astype(int))),
    "descriptions": dict(zip(desc_df["Disease"], desc_df["Description"])),
    "precautions": {
        row["Disease"]: [
            str(row[c]).strip().capitalize() for c in prec_df.columns[1:] if pd.notna(row[c])
        ]
        for _, row in prec_df.iterrows()
    },
}
MODELS.mkdir(exist_ok=True)
joblib.dump(artifacts, MODELS / "disease_model.pkl")
print("\nSaved to models/disease_model.pkl")