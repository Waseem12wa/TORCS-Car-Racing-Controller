import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import joblib


if __name__ == '__main__':

    data = pd.read_csv(r'reduced.csv')
    data.dropna()

    x = data.iloc[:, :-2].values
    y = data.iloc[:, 34:35].values.ravel()

    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print("Started Training")
    gnb = ExtraTreesClassifier(max_features=None,verbose=0,n_jobs=-1)
    gnb.fit(x_train,y_train)
    joblib.dump(gnb, 'accel.pkl')
    print("Successfully Trained")
    print(gnb.classes_)

    y_pred_test = gnb.predict(x_test)
    acc = accuracy_score(y_test, y_pred_test)
    print("accuracy_score : "+ str(acc))