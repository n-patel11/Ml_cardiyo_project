# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# import pandas as pd

# app = Flask(__name__)

# # ---------------------------
# # Decision Tree Classes
# # ---------------------------

# class Node:
#     def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value


# class DecisionTreeScratch:
#     def __init__(self, max_depth=5):
#         self.max_depth = max_depth

#     def gini(self, y):
#         m = len(y)
#         if m == 0:
#             return 0
#         p1 = np.sum(y) / m
#         p0 = 1 - p1
#         return 1 - (p0 ** 2 + p1 ** 2)

#     def best_split(self, X, y):
#         best_gini = 1
#         best_feature = None
#         best_threshold = None

#         n_samples, n_features = X.shape
#         for feature in range(n_features):
#             thresholds = np.unique(X[:, feature])
#             for threshold in thresholds:
#                 left_idx = X[:, feature] <= threshold
#                 right_idx = X[:, feature] > threshold

#                 if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
#                     continue

#                 gini_left = self.gini(y[left_idx])
#                 gini_right = self.gini(y[right_idx])

#                 weighted_gini = (
#                     len(y[left_idx]) / n_samples * gini_left
#                     + len(y[right_idx]) / n_samples * gini_right
#                 )

#                 if weighted_gini < best_gini:
#                     best_gini = weighted_gini
#                     best_feature = feature
#                     best_threshold = threshold

#         return best_feature, best_threshold

#     def build_tree(self, X, y, depth=0):
#         if len(np.unique(y)) == 1:
#             return Node(value=y[0])

#         if depth >= self.max_depth:
#             return Node(value=np.round(np.mean(y)))

#         feature, threshold = self.best_split(X, y)

#         if feature is None:
#             return Node(value=np.round(np.mean(y)))

#         left_idx = X[:, feature] <= threshold
#         right_idx = X[:, feature] > threshold

#         left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
#         right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

#         return Node(feature, threshold, left, right)

#     def fit(self, X, y):
#         self.root = self.build_tree(X, y)

#     def predict_sample(self, x, node):
#         if node.value is not None:
#             return node.value
#         if x[node.feature] <= node.threshold:
#             return self.predict_sample(x, node.left)
#         else:
#             return self.predict_sample(x, node.right)

#     def predict(self, X):
#         return np.array([self.predict_sample(x, self.root) for x in X])

#     # ✅ NEW METHOD (for explanation)
#     def predict_with_path(self, x):
#         node = self.root
#         path = []

#         while node.value is None:
#             feature = node.feature
#             threshold = node.threshold

#             if x[feature] <= threshold:
#                 path.append((feature, threshold, "<="))
#                 node = node.left
#             else:
#                 path.append((feature, threshold, ">"))
#                 node = node.right

#         return node.value, path


# # ---------------------------
# # Load or Train Model
# # ---------------------------

# try:
#     with open("cardio_model.pkl", "rb") as f:
#         tree = pickle.load(f)
#     print("Model loaded successfully!")
# except FileNotFoundError:
#     print("Training model...")
#     df = pd.read_csv("cardio_train.csv", sep=";")
#     df = df.drop("id", axis=1)
#     df["age"] = df["age"] / 365
#     df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
#     df = df[(df["ap_hi"] < 250) & (df["ap_lo"] < 200)]

#     X = df.drop("cardio", axis=1).values
#     y = df["cardio"].values

#     split = int(0.8 * len(X))
#     X_train = X[:split]
#     y_train = y[:split]

#     tree = DecisionTreeScratch(max_depth=5)
#     tree.fit(X_train, y_train)

#     with open("cardio_model.pkl", "wb") as f:
#         pickle.dump(tree, f)

#     print("Model trained and saved!")


# # Feature Names
# feature_names = [
#     "Age",
#     "Gender",
#     "Height",
#     "Weight",
#     "Systolic BP",
#     "Diastolic BP",
#     "Cholesterol",
#     "Glucose",
#     "Smoking",
#     "Alcohol",
#     "Physical Activity"
# ]


# # ---------------------------
# # Flask Routes
# # ---------------------------

# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/form")
# def form():
#     return render_template("form.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     age = float(request.form["age"])
#     gender = int(request.form["gender"])
#     height = float(request.form["height"])
#     weight = float(request.form["weight"])
#     ap_hi = float(request.form["ap_hi"])
#     ap_lo = float(request.form["ap_lo"])
#     cholesterol = int(request.form["cholesterol"])
#     gluc = int(request.form["gluc"])
#     smoke = int(request.form["smoke"])
#     alco = int(request.form["alco"])
#     active = int(request.form["active"])

#     X = np.array([[age, gender, height, weight, ap_hi, ap_lo,
#                    cholesterol, gluc, smoke, alco, active]])

#     # Prediction with explanation
#     prediction, path = tree.predict_with_path(X[0])

#     risk_reasons = []

#     for feature, threshold, condition in path:
#         value = X[0][feature]
#         if condition == ">" and value > threshold:
#             risk_reasons.append(f"{feature_names[feature]} is high ({value})")

#     if prediction == 1:
#         result_text = "High risk of cardiovascular disease! Please consult a doctor."
#     else:
#         result_text = "Low risk of cardiovascular disease. Keep maintaining healthy habits!"
#         risk_reasons = []  # clear reasons for low risk

#     return render_template(
#         "form.html",
#         prediction_text=result_text,
#         risk_reasons=risk_reasons
#     )


# # if __name__ == "__main__":
# #     app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------
# Decision Tree Classes
# ---------------------------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba


class DecisionTreeScratch:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        p1 = np.sum(y) / m
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)

    def best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gini_left = self.gini(y[left_idx])
                gini_right = self.gini(y[right_idx])

                weighted_gini = (
                    len(y[left_idx]) / n_samples * gini_left +
                    len(y[right_idx]) / n_samples * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):

        if len(np.unique(y)) == 1:
            return Node(value=y[0], proba=float(y[0]))

        if depth >= self.max_depth:
            val = np.round(np.mean(y))
            proba = np.mean(y)
            return Node(value=val, proba=proba)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            val = np.round(np.mean(y))
            proba = np.mean(y)
            return Node(value=val, proba=proba)

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right = self.build_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)

        return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])

    def predict_with_path(self, x):

        node = self.root
        path = []

        while node.value is None:

            feature = node.feature
            threshold = node.threshold

            if x[feature] <= threshold:
                path.append((feature, threshold, "<="))
                node = node.left
            else:
                path.append((feature, threshold, ">"))
                node = node.right

        return node.value, node.proba, path


# ---------------------------
# Train Model Function
# ---------------------------

def train_model():

    print("Training model...")

    df = pd.read_csv("cardio_train.csv", sep=";")

    df = df.drop("id", axis=1)

    df["age"] = df["age"] / 365

    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[(df["ap_hi"] < 250) & (df["ap_lo"] < 200)]

    X = df.drop("cardio", axis=1).values
    y = df["cardio"].values

    split = int(0.8 * len(X))

    X_train = X[:split]
    y_train = y[:split]

    tree = DecisionTreeScratch(max_depth=5)
    tree.fit(X_train, y_train)

    with open("cardio_model.pkl", "wb") as f:
        pickle.dump(tree, f)

    print("Model trained and saved!")

    return tree


# ---------------------------
# Load Model Safely
# ---------------------------

MODEL_PATH = "cardio_model.pkl"

if os.path.exists(MODEL_PATH):

    try:
        with open(MODEL_PATH, "rb") as f:
            tree = pickle.load(f)

        print("Model loaded successfully!")

    except Exception as e:

        print("Model load failed. Retraining...", e)
        tree = train_model()

else:

    tree = train_model()


# ---------------------------
# Feature Names
# ---------------------------

feature_names = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "Systolic BP",
    "Diastolic BP",
    "Cholesterol",
    "Glucose",
    "Smoking",
    "Alcohol",
    "Physical Activity"
]


# ---------------------------
# Flask Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():

    patient_data = {
        "Age": int(float(request.form["age"])),
        "Gender": "Male" if int(request.form["gender"]) == 2 else "Female",
        "Height (cm)": float(request.form["height"]),
        "Weight (kg)": float(request.form["weight"]),
        "Systolic BP": float(request.form["ap_hi"]),
        "Diastolic BP": float(request.form["ap_lo"]),
        "Cholesterol": int(request.form["cholesterol"]),
        "Glucose": int(request.form["gluc"]),
        "Smoking": "Yes" if int(request.form["smoke"]) == 1 else "No",
        "Alcohol": "Yes" if int(request.form["alco"]) == 1 else "No",
        "Physical Activity": "Yes" if int(request.form["active"]) == 1 else "No"
    }

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    ap_hi = float(request.form["ap_hi"])
    ap_lo = float(request.form["ap_lo"])
    cholesterol = int(request.form["cholesterol"])
    gluc = int(request.form["gluc"])
    smoke = int(request.form["smoke"])
    alco = int(request.form["alco"])
    active = int(request.form["active"])

    X = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                   cholesterol, gluc, smoke, alco, active]])

    prediction, proba, path = tree.predict_with_path(X[0])

    confidence_pct = round((proba if prediction == 1 else 1 - proba) * 100)
    risk_prob_pct = round(proba * 100)
    
    # Calculate simple health score
    health_score = 100 - (risk_prob_pct * 0.8) # Base score minus risk penalty
    
    # Deduct points for obvious risk factors
    detected_risks = []
    if cholesterol > 1:
        detected_risks.append("High Cholesterol")
        health_score -= 5
    if ap_hi >= 130 or ap_lo >= 80:
        detected_risks.append("High Blood Pressure")
        health_score -= 5
    if smoke == 1:
        detected_risks.append("Smoking Habit")
        health_score -= 5
    if active == 0:
        detected_risks.append("Lack of Physical Activity")
        health_score -= 5
        
    health_score = max(0, min(100, round(health_score)))

    risk_reasons = []

    for feature, threshold, condition in path:

        value = X[0][feature]

        if condition == ">" and value > threshold:
            risk_reasons.append(f"{feature_names[feature]} is high ({value})")

    # Recommendations based on inputs
    recommendations = []
    if active == 0: recommendations.append("Exercise at least 30 minutes daily")
    if cholesterol > 1: recommendations.append("Eat low-cholesterol food")
    if smoke == 1 or alco == 1: recommendations.append("Avoid smoking and alcohol")
    bmi = weight / ((height/100)**2)
    if bmi > 25: recommendations.append("Maintain healthy weight")
    recommendations.append("Schedule regular health checkups")

    if prediction == 1:
        result_text = "⚠️ High Risk of Cardiovascular Disease Detected. Please consult a doctor and improve lifestyle habits."
        risk_level = "High"
    else:
        result_text = "✅ Low Risk of Cardiovascular Disease. Maintain a healthy lifestyle and regular checkups."
        risk_level = "Low"
        risk_reasons = []

    if 50 <= risk_prob_pct <= 70 and prediction == 0:
         risk_level = "Medium"

    return render_template(
        "result.html",
        prediction=prediction,
        prediction_text=result_text,
        risk_reasons=risk_reasons,
        risk_level=risk_level,
        confidence_pct=confidence_pct,
        risk_prob_pct=risk_prob_pct,
        health_score=health_score,
        patient_data=patient_data,
        detected_risks=detected_risks,
        recommendations=recommendations,
        bmi=round(bmi, 1)
    )


# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)