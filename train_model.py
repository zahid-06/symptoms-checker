import re

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

# Fixes for spelling and spacing problems in the Kaggle files
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
    text = text.strip().replace(" ", "_")
    return re.sub(r"_+", "_", text)


def clean_disease(text):
    text = re.sub(r"\s+", " ", text.strip())
    return DISEASE_RENAME.get(text, text)


# 1. Load the four CSV files
data = pd.read_csv("dataset.csv")
severity_df = pd.read_csv("Symptom-severity.csv")
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")

symptom_cols = [c for c in data.columns if c.startswith("Symptom")]

# 2. Clean the text
data["Disease"] = data["Disease"].map(clean_disease)
for col in symptom_cols:
    data[col] = data[col].map(clean_symptom)

severity_df["Symptom"] = severity_df["Symptom"].map(clean_symptom)
severity_df["Symptom"] = severity_df["Symptom"].replace(
    {"foul_smell_ofurine": "foul_smell_of_urine"}
)
severity_df = severity_df[severity_df["Symptom"] != "prognosis"]

desc_df["Disease"] = desc_df["Disease"].map(clean_disease)
prec_df["Disease"] = prec_df["Disease"].map(clean_disease)

# 3. Turn each row into a 0/1 vector (one column per symptom)
all_symptoms = sorted(
    {s for s in data[symptom_cols].values.ravel() if isinstance(s, str)}
)
X = pd.DataFrame(0, index=data.index, columns=all_symptoms)
for col in symptom_cols:
    for row, symptom in data[col].items():
        if isinstance(symptom, str):
            X.loc[row, symptom] = 1
y = data["Disease"]

print(f"Rows: {len(X)} | Diseases: {y.nunique()} | Symptoms: {len(all_symptoms)}")

# 4. Remove duplicate rows. The file has many identical rows, and testing on
# copies of training rows would give a fake 100% accuracy.
combined = pd.concat([X, y], axis=1).drop_duplicates()
X = combined.drop(columns="Disease")
y = combined["Disease"]
print(f"Unique rows after removing duplicates: {len(X)}")

# 5. Compare three models with 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": BernoulliNB(),
}
print("\n--- Cross-validation accuracy ---")
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 6. Train the final model on all unique rows
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X, y)

# 7. Save everything the API needs
artifacts = {
    "model": final_model,
    "symptoms": all_symptoms,
    "severity": dict(zip(severity_df["Symptom"], severity_df["weight"].astype(int))),
    "descriptions": dict(zip(desc_df["Disease"], desc_df["Description"])),
    "precautions": {
        row["Disease"]: [
            str(row[c]).strip().capitalize()
            for c in prec_df.columns[1:]
            if pd.notna(row[c])
        ]
        for _, row in prec_df.iterrows()
    },
}
joblib.dump(artifacts, "disease_model.pkl")
print("\nSaved to disease_model.pkl")
