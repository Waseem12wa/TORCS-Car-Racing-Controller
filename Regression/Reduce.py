import pandas as pd

if __name__ == '__main__':

    data = pd.read_csv(r'dataset.csv')
    data = data.dropna()
    corr = data.corr()
    features = data.columns.values
    opponent = features[8:44]
    focus = features[74:79]
    newd = data.drop(columns=opponent)
    newd = newd.drop(columns=focus)
    newd = newd.drop(columns=['damage'])
    newd = newd.drop(columns=['fuel'])
    newd = newd.drop(columns=['lastLapTime'])
    newd = newd.drop(columns=['racePos'])
    newd.to_csv('datareduced.csv', index=False, header=True)
