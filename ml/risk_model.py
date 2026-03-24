# =============================================================
# ml/risk_model.py
# Student Success Copilot — ML Component
#
# WHAT THIS FILE DOES:
#   1. Generates a synthetic dataset of 300 fake students
#   2. Trains a Decision Tree to predict risk level
#   3. Evaluates the model (accuracy + F1 score)
#   4. Provides predict_risk() — takes a student, returns risk + explanation
#
# WHY A DECISION TREE?
#   It's the most explainable ML model. Every prediction has a clear
#   reason: "HIGH risk because deadline <= 3 AND stress > 6".
#   That satisfies the coursework's "explain your recommendations" requirement.
# =============================================================

import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Fix random seed so results are the same every run (important for demos)
random.seed(42)
np.random.seed(42)


# -------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC DATASET
# -------------------------------------------------------------
# We invent 300 "past students" with realistic patterns.
# Each student has features (inputs) and a label (the answer we want to predict).
#
# Features:
#   days_until_deadline  — how many days until their next major deadline (1–14)
#   stress_level         — self-reported stress on a scale of 1–10
#   confidence           — self-reported confidence on a scale of 1–10
#   hours_available      — study hours available this week (2–20)
#   missed_sessions      — how many sessions they've skipped (0–5)
#   gender_encoded       — 0 = female, 1 = male, 2 = other (encoded as number)
#
# Label: risk_level — "Low", "Medium", or "High"
#
# The patterns we build in are realistic:
#   High risk  = tight deadline + high stress + low confidence
#   Low risk   = plenty of time + low stress + high confidence
#   Medium     = everything in between

def generate_dataset(n=300):
    """
    Creates a list of student dictionaries and a matching list of risk labels.
    Returns: (features_list, labels_list)
    """
    features = []
    labels = []

    for _ in range(n):
        days   = random.randint(1, 14)
        stress = random.randint(1, 10)
        conf   = random.randint(1, 10)
        hours  = random.randint(2, 20)
        missed = random.randint(0, 5)
        gender = random.randint(0, 2)

        # --- Determine risk label based on realistic rules ---
        # These rules define the "ground truth" in our synthetic world.
        # The ML model will learn to approximate these patterns from data.
        if days <= 3 and stress >= 7:
            risk = "High"
        elif days <= 3 and conf <= 4:
            risk = "High"
        elif missed >= 3 and stress >= 6:
            risk = "High"
        elif days <= 5 and stress >= 5 and conf <= 5:
            risk = "Medium"
        elif missed >= 2 and conf <= 5:
            risk = "Medium"
        elif hours <= 5 and days <= 7:
            risk = "Medium"
        else:
            risk = "Low"

        # Add a tiny amount of noise (10% chance of wrong label)
        # This makes the dataset realistic — real data is never perfectly clean
        if random.random() < 0.10:
            risk = random.choice(["Low", "Medium", "High"])

        features.append([days, stress, conf, hours, missed, gender])
        labels.append(risk)

    return features, labels


# -------------------------------------------------------------
# STEP 2: TRAIN THE MODEL
# -------------------------------------------------------------
# We split data into:
#   80% training   — the model LEARNS from this
#   20% testing    — we CHECK if it actually learned (never used in training)
#
# This split is crucial. Without it, the model might just memorise the answers
# (called "overfitting") without actually understanding the pattern.

def train_model():
    """
    Generates data, splits it, trains a Decision Tree, evaluates it.
    Returns the trained model and the feature column names.
    """
    print("=" * 55)
    print("  ML COMPONENT — Student Risk Predictor")
    print("=" * 55)

    # Generate our synthetic dataset
    X, y = generate_dataset(n=300)
    print(f"\n[1] Dataset generated: {len(X)} students")

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,    # 20% goes to testing
        random_state=42    # same split every run
    )
    print(f"[2] Split: {len(X_train)} training, {len(X_test)} testing")

    # Create and train the Decision Tree
    # max_depth=4 keeps the tree readable — not too deep, not too shallow
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)  # <-- this is where the learning happens
    print("[3] Model trained (Decision Tree, max depth 4)")

    # Evaluate on the TEST set (data the model has never seen)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[4] EVALUATION RESULTS (on unseen test data):")
    print(f"    Accuracy : {acc:.2%}")
    print(f"    F1 Score : {f1:.2f}")
    print()
    print(classification_report(y_test, y_pred))

    # Print the tree in text form — useful for your report
    feature_names = [
        "days_until_deadline",
        "stress_level",
        "confidence",
        "hours_available",
        "missed_sessions",
        "gender_encoded"
    ]
    print("[5] Decision tree structure (first 3 levels):")
    tree_text = export_text(model, feature_names=feature_names, max_depth=3)
    print(tree_text)

    return model, feature_names


# -------------------------------------------------------------
# STEP 3: PREDICT + EXPLAIN
# -------------------------------------------------------------
# This is the function the rest of your system will call.
# It takes a student's data and returns:
#   - the risk level ("Low", "Medium", or "High")
#   - a confidence percentage
#   - a human-readable explanation of WHY

def predict_risk(model, student: dict) -> dict:
    """
    Predicts risk level for a single student.

    Parameters:
        model   — the trained Decision Tree (from train_model())
        student — a dict with keys:
                    days_until_deadline (int)
                    stress_level        (int 1-10)
                    confidence          (int 1-10)
                    hours_available     (int)
                    missed_sessions     (int)
                    gender              (str: "female"/"male"/"other")

    Returns:
        dict with keys: risk, confidence_pct, explanation, factors
    """
    # Encode gender as a number (ML models need numbers, not strings)
    gender_map = {"female": 0, "male": 1, "other": 2}
    gender_encoded = gender_map.get(student.get("gender", "other"), 2)

    # Build the feature vector (must be in the same order as training)
    features = [[
        student["days_until_deadline"],
        student["stress_level"],
        student["confidence"],
        student["hours_available"],
        student["missed_sessions"],
        gender_encoded
    ]]

    # Get prediction and probability
    risk = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence_pct = round(max(proba) * 100, 1)

    # Build a human-readable explanation by checking which factors are critical
    factors = []
    d = student["days_until_deadline"]
    s = student["stress_level"]
    c = student["confidence"]
    m = student["missed_sessions"]
    h = student["hours_available"]

    if d <= 3:
        factors.append(f"deadline is very close ({d} day(s) away)")
    if s >= 7:
        factors.append(f"stress level is high ({s}/10)")
    if c <= 4:
        factors.append(f"confidence is low ({c}/10)")
    if m >= 2:
        factors.append(f"missed {m} session(s) recently")
    if h <= 5:
        factors.append(f"limited study time available ({h} hours/week)")

    if factors:
        explanation = f"Predicted {risk} risk because: " + "; ".join(factors) + "."
    else:
        explanation = f"Predicted {risk} risk. No major warning signals detected."

    return {
        "risk":           risk,
        "confidence_pct": confidence_pct,
        "explanation":    explanation,
        "factors":        factors
    }


# -------------------------------------------------------------
# STEP 4: LIMITATIONS (required by coursework)
# -------------------------------------------------------------
LIMITATIONS = """
MODEL LIMITATIONS:
  1. Synthetic data — trained on invented students, not real ones.
     Real patterns may differ significantly.
  2. Small dataset — 300 students is very small for ML.
     More data = more reliable patterns.
  3. No temporal features — doesn't track how stress changes over time.
  4. Gender is included as a feature; this could introduce bias
     if the model learns to treat genders differently for risk.
     In production, this would need a fairness audit.
  5. Decision Tree can overfit — even with max_depth=4, it may
     be too specific to the training data.
"""


# -------------------------------------------------------------
# RUN THIS FILE DIRECTLY TO TEST EVERYTHING
# -------------------------------------------------------------
if __name__ == "__main__":
    # Train and evaluate the model
    model, feature_names = train_model()

    print("\n" + "=" * 55)
    print("  PREDICTION DEMO")
    print("=" * 55)

    # Test student 1: high risk scenario
    student_a = {
        "days_until_deadline": 2,
        "stress_level":        9,
        "confidence":          3,
        "hours_available":     4,
        "missed_sessions":     3,
        "gender":              "male"
    }

    # Test student 2: low risk scenario
    student_b = {
        "days_until_deadline": 10,
        "stress_level":        3,
        "confidence":          8,
        "hours_available":     15,
        "missed_sessions":     0,
        "gender":              "female"
    }

    for name, student in [("Student A (at-risk)", student_a),
                           ("Student B (safe)",   student_b)]:
        result = predict_risk(model, student)
        print(f"\n{name}:")
        print(f"  Risk level  : {result['risk']}")
        print(f"  Confidence  : {result['confidence_pct']}%")
        print(f"  Explanation : {result['explanation']}")

    print("\n" + LIMITATIONS)
