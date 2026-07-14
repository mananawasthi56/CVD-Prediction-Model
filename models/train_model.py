import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from preprocessing.feature_engineering import feature_engineering


def train_models():

    X_train, X_test, y_train, y_test = feature_engineering()

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=300,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss"
        )

    }

    best_model = None
    best_accuracy = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        print(f"{name} : {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    joblib.dump(best_model, "models/best_model.pkl")

    print("\nBest Model Saved Successfully")