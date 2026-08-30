# ==========================================
# MODULE 3 - AI CACHE MISS PREDICTOR
# ==========================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("======================================")
print("       MODULE 3 - AI CACHE PREDICTOR")
print("======================================")


# ==========================================
# 1. FIND PROJECT FOLDER
# ==========================================

project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================
# 2. CACHE RESULTS FILE
# ==========================================

input_file = os.path.join(
    project_folder,
    "data",
    "cache_results.csv"
)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading cache results...")

data = pd.read_csv(input_file)

print(
    f"Loaded {len(data)} cache records."
)


# ==========================================
# 4. CREATE AI FEATURES
# ==========================================

print("\nCreating AI features...")


# Feature 1: Memory Address

data["memory_address"] = data[
    "memory_address"
].astype(int)


# Feature 2: Operation
# READ  = 0
# WRITE = 1

data["operation_code"] = data[
    "operation"
].map({
    "READ": 0,
    "WRITE": 1
})


# Feature 3: Previous Memory Address

data["previous_address"] = data[
    "memory_address"
].shift(1)


# First access has no previous address

data["previous_address"] = data[
    "previous_address"
].fillna(
    data["memory_address"]
)


# Feature 4: Address Difference

data["address_difference"] = (
    data["memory_address"]
    - data["previous_address"]
).abs()


# Feature 5: Access Frequency

data["access_frequency"] = (
    data.groupby(
        "memory_address"
    ).cumcount()
)


# ==========================================
# 5. CREATE TARGET
# ==========================================

# MISS = 0
# HIT  = 1

data["target"] = data[
    "cache_result"
].map({
    "MISS": 0,
    "HIT": 1
})


print("AI features created successfully!")


# ==========================================
# 6. SELECT FEATURES
# ==========================================

features = [
    "memory_address",
    "operation_code",
    "previous_address",
    "address_difference",
    "access_frequency"
]


X = data[features]

y = data["target"]


print("\nFeatures used by AI:")

for feature in features:
    print("-", feature)


# ==========================================
# 7. SPLIT DATASET
# ==========================================

print("\nSplitting dataset...")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ==========================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest AI model...")


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 9. TRAIN AI MODEL
# ==========================================

print("Training AI model...")


model.fit(
    X_train,
    y_train
)


print("AI training completed!")


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")


predictions = model.predict(
    X_test
)


# ==========================================
# 11. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


# ==========================================
# 12. DISPLAY AI RESULTS
# ==========================================

print("\n======================================")
print("          AI MODEL RESULTS")
print("======================================")


print(
    f"AI Accuracy : {accuracy * 100:.2f}%"
)


# ==========================================
# 13. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")


print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "MISS",
            "HIT"
        ],
        zero_division=0
    )
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")


matrix = confusion_matrix(
    y_test,
    predictions
)


print(matrix)


# ==========================================
# 15. FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")


importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print(importance)


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("       AI PREDICTION COMPLETED")
print("======================================")

print("\nModel successfully trained!")
print("Cache HIT/MISS prediction completed!")