import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN


def main():
    features_sentiment = joblib.load("models/sentiment_features.sav")
    target_sentiment = joblib.load("models/sentiment_target.sav")
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

    features_sentiment, target_sentiment = ADASYN(random_state=0).fit_resample(features_sentiment, target_sentiment)
    clf.fit(features_sentiment, target_sentiment)
    joblib.dump(clf, "models/sentiment_model_clf.sav")

    features_fakereview, target_fakereview = ADASYN(random_state=0).fit_resample(features_fakereview, target_fakereview)
    clf.fit(features_fakereview, target_fakereview)
    joblib.dump(clf, "models/fakereview_model_clf.sav")


if __name__ == "__main__":
    main()
