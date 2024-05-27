import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def main():
    data = joblib.load("models/sentiment_data.sav")
    # clf = RandomForestClassifier(n_jobs=-1)
    clf = LogisticRegression(solver="liblinear")
    # clf = LinearSVC()
    # clf = GaussianNB()

    label = "is_bad_review"
    ignore_cols = [label, "review", "review_clean", "rating"]
    features = [c for c in data.columns if c not in ignore_cols]
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[label], test_size=0.2)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
