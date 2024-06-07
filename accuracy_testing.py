import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # features_sentiment = joblib.load("models/sentiment_features.sav")
    # target_sentiment = joblib.load("models/sentiment_target.sav")
    features_fakereview = joblib.load("models/fakereview_features.sav")
    target_fakereview = joblib.load("models/fakereview_target.sav")

    clf = LogisticRegression(
        solver="liblinear",
        class_weight={
            0: 0.50,
            1: 0.50
        },
        random_state=0
    )

    # x_train, x_test, y_train, y_test = train_test_split(features_sentiment, target_sentiment, test_size=0.2)
    # x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print("SENTIMENT:")
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    x_train, x_test, y_train, y_test = train_test_split(features_fakereview, target_fakereview, test_size=0.2, random_state=0)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("FAKEREVIEW:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
