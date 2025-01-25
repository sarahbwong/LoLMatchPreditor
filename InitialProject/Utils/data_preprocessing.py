import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_combine_data(file_paths):
    dfs = []
    for i, file in enumerate(file_paths):
        percent = str((i + 1) * 20)
        df = pd.read_csv(file)
        df['percent'] = percent
        df['percent_encoded'] = i
        dfs.append(df)
    df = pd.concat(dfs)

    df.drop(columns=[
        'fullTimeMS', 'timePercent', 'Unnamed: 0', 'matchID',
        'blueTotalGold', 'blueAvgPlayerLevel', 'redFirstBlood', 
        'redRiftHeraldKill', 'redTotalGold', 'redAvgPlayerLevel', 'redWin'
    ], inplace=True)

    df['blueFirstBlood'] = df['blueFirstBlood'].astype(int)
    df['blueWin'] = df['blueWin'].astype(int)
    df = df.reset_index()
    return df

def split_and_scale_data(df, test_size, random_state):
    y_win = df['blueWin'].copy()
    X = df.drop(columns=['blueWin', 'percent', 'percent_encoded'], axis=1)
    y = df['percent_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #Theres something really wrong with this i cant even figure it out
    y_train_win = y_win.loc[X_train.index].astype(int)
    y_test_win = y_win.loc[X_test.index].astype(int)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, y_train_win, y_test_win

def set_b_features(X):
    """
    Make a new set of X data by doing algorithmic feature engineering
    """

    # First, we make a copy of the original X features
    X_new = X.copy()

    # ----------------------------------------------------------
    # Add interaction terms
    X_new["blueChampionKill * blueDragonKill"] = X["blueChampionKill"] * X["blueDragonKill"]
    X_new["blueFirstBlood * blueBaronKill"] = X["blueFirstBlood"] * X["blueBaronKill"]
    X_new["blueFirstBlood * redBaronKill"] = X["blueFirstBlood"] * X["redBaronKill"]
    X_new["blueMinionsKilled * redChampionKill"] = X["blueMinionsKilled"] * X["redChampionKill"]
    X_new["redChampionKill * redDragonKill"] = X["redChampionKill"] * X["redDragonKill"]
    X_new["redChampionKill * redTowerKill"] = X["redChampionKill"] * X["redTowerKill"]
    X_new["redDragonHextechKill * redTowerKill"] = X["redDragonHextechKill"] * X["redTowerKill"]
    X_new["redDragonChemtechKill * redTowerKill"] = X["redDragonChemtechKill"] * X["redTowerKill"]
    X_new["redDragonFireKill * redTowerKill"] = X["redDragonFireKill"] * X["redTowerKill"]
    X_new["redDragonAirKill * redTowerKill"] = X["redDragonAirKill"] * X["redTowerKill"]
    X_new["redDragonEarthKill * redTowerKill"] = X["redDragonEarthKill"] * X["redTowerKill"]
    X_new["redDragonWaterKill * redTowerKill"] = X["redDragonWaterKill"] * X["redTowerKill"]
    # ----------------------------------------------------------
    return X_new

def set_c_features(df):
    """
    Extract differential features from the dataset.
    """
    new_df = pd.DataFrame()
    new_df['ChampionKill'] = df['blueChampionKill'] - df['redChampionKill']
    new_df['FirstBlood'] = df['blueFirstBlood']
    new_df['DragonKill'] = df['blueDragonKill'] - df['redDragonKill']
    new_df['DragonAirKill'] = df['blueDragonAirKill'] - df['redDragonAirKill']
    new_df['DragonEarthKill'] = df['blueDragonEarthKill'] - df['redDragonEarthKill']
    new_df['DragonWaterKill'] = df['blueDragonWaterKill'] - df['redDragonWaterKill']
    new_df['DragonHextechKill'] = df['blueDragonHextechKill'] - df['redDragonHextechKill']
    new_df['DragonFireKill'] = df['blueDragonFireKill'] - df['redDragonFireKill']
    new_df['DragonChemtechKill'] = df['blueDragonChemtechKill'] - df['redDragonChemtechKill']
    new_df['DragonElderKill'] = df['blueDragonElderKill'] - df['redDragonElderKill']
    new_df['BaronKill'] = df['blueBaronKill'] - df['redBaronKill']
    new_df['TowerKill'] = df['blueTowerKill'] - df['redTowerKill']
    new_df['InhibitorKill'] = df['blueInhibitorKill'] - df['redInhibitorKill']
    new_df['MinionsKilled'] = df['blueMinionsKilled'] - df['redMinionsKilled']
    new_df['JungleMinionsKilled'] = df['blueJungleMinionsKilled'] - df['redJungleMinionsKilled']
    return new_df
