from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

print("Test accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "model.pkl")
