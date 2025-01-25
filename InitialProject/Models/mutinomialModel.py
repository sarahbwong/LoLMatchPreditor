import pandas as pd 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

scaler = StandardScaler()

# read csv files and create a column named percent to indicate which file 
# percentage of time passed in game the data belongs to
# also add a percent_encoded column to create a multinomial logistic regression model
df_20 = pd.read_csv('/content/LoLytics/InitialProject/Datasets/full_data_20.csv')
df_20['percent'] = "20"
df_20['percent_encoded'] = 0
df_40 = pd.read_csv('/content/LoLytics/InitialProject/Datasets/full_data_40.csv')
df_40['percent'] = "40"
df_40['percent_encoded'] = 1
df_60 = pd.read_csv('/content/LoLytics/InitialProject/Datasets/full_data_60.csv')
df_60['percent'] = "60"
df_60['percent_encoded'] = 2
df_80 = pd.read_csv('/content/LoLytics/InitialProject/Datasets/full_data_80.csv')
df_80['percent'] = "80"
df_80['percent_encoded'] = 3
df_100 = pd.read_csv('/content/LoLytics/InitialProject/Datasets/full_data_100.csv')
df_100['percent'] = "100"
df_100['percent_encoded'] = 4

df = pd.concat([df_20, df_40, df_60, df_80, df_100])
### remove levels and gold because these are influenced by kills and objectives
### remove any features that mention red rift herald because the other things (dragon, baron) will respawn
df.drop(columns=['fullTimeMS','timePercent', 'Unnamed: 0', 'matchID', 'blueTotalGold', 'blueAvgPlayerLevel',
       'redFirstBlood','redRiftHeraldKill', 'redTotalGold', 'redAvgPlayerLevel', 'redWin'], inplace=True)
df['blueFirstBlood'] = df['blueFirstBlood'].astype(int) # converting boolean as type int
df['blueWin'] = df['blueWin'].astype(int) # converting boolean as type int

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['blueWin', 'percent', 'percent_encoded'], axis=1), 
                                                    df['percent_encoded'],
                                                     test_size=0.30, random_state=5)

y_test_win = df_20.loc[X_test.index, 'blueWin'].astype(int)
y_train_win = df_20.loc[X_train.index, 'blueWin'].astype(int)

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

### Training the multinomial logistic regression model
X_train_scaled = sm.add_constant(X_train_scaled) # Add an intercept to the training set
multilog = sm.MNLogit(y_train, X_train_scaled) # Fit the multinomial logistic regression model
result = multilog.fit(maxiter=30) # printing result will give information about model - not necessary at this stage

### Using model to predict probabilities of the data belonging to a game at stage 20%, 40%, 60%, 80%, or 100% of game passed
# The prediction is for 0,1,2,3,4 which represent 20%, 40%, 60%, 80%, 100% respectively
# Add an intercept to the test set
X_test_scaled = sm.add_constant(X_test_scaled)
predicted_probabilities = result.predict(X_test_scaled) # Get predicted probabilities for the test set
probabilities_df = pd.DataFrame(predicted_probabilities)
pd.set_option('display.float_format', '{:.10f}'.format) # fix formatting

# Do we need to test the model for accuracy? 
# We aren't really getting a classification, just probabilities of belong to each of the 5 classes

#Initializing list of win probabilities for each time stamp dict. I.e. given we are in stamp x, what is the probability that blue wins for that
#specific match.
win_prob_stamp = {}

#Iterating through all the time stamps and fitting models for each stamp of the game and predicting game outcome.
for stamp in range(5):
    stamp_index = y_train == stamp
    X_train_stamp = X_train_scaled[stamp_index]
    y_train_stamp_win = y_train_win[stamp_index]

    model = LogisticRegression()
    #model = XGBClassifier()
    model.fit(X_train_stamp, y_train_stamp_win)

    win_prob_stamp[stamp] = model.predict_proba(X_test_scaled)[:, 1]


win_prob_df = pd.DataFrame(win_prob_stamp, index=X_test_scaled.index)

overall_win_prob = (win_prob_df * probabilities_df).sum(axis=1)
overall_result = (overall_win_prob > 0.5).astype(int)

#Calculating overall model classification metrics for all stamps (Accuracy, recall etc.)
accuracy = accuracy_score(y_test_win, overall_result)
print(f"Accuracy: {accuracy:.2f}")

precision = precision_score(y_test_win, overall_result)
recall = recall_score(y_test_win, overall_result)
f1 = f1_score(y_test_win, overall_result)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

fpr, tpr, thresholds = roc_curve(y_test_win, overall_win_prob)
roc_auc = roc_auc_score(y_test_win, overall_win_prob)

#Plotting ROC Curve for overall win probability
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for Overall Win Probability Model')
plt.grid(True)
plt.savefig('roc_curve.png')

print(f"AUC: {roc_auc:.2f}")

#Classification report for each time stamp to get insight as how the model performs for each stamp and not just its entirety
for stamp in range(5):
    stamp_index = y_test == stamp
    y_true = y_test_win[stamp_index]       
    y_pred = overall_result[stamp_index]   
    
    #https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html check documentation -- example used from there
    print(f"Report for TimeStamp {(stamp + 1) * 20}%:")
    print(classification_report(y_true, y_pred))
    #Here we can see that the model performs best for the 20% stamp and worst for the 100% stamp.