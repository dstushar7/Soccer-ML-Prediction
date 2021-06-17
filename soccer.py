from flask import Flask, render_template
from flask import request
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from pprint import pprint
app = Flask(__name__)

@app.route('/')
def runpy():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def form_post():
    global team1
    global team2
    team1 = request.form['sel1']
    team2 = request.form['sel2']
    if team1 != team2 :
        data = pd.read_csv('dataset.csv')
        data = data[data.MW > 3]
        teamname = team1

        data.drop(['Unnamed: 0','HomeTeam', 'AwayTeam', 'Date', 'MW', 'HTFormPtsStr', 'ATFormPtsStr', 'FTHG', 'FTAG',
           'HTGS', 'ATGS', 'HTGC', 'ATGC','HomeTeamLP', 'AwayTeamLP','DiffPts','HTFormPts','ATFormPts',
           'HM4','HM5','AM4','AM5','HTLossStreak5','ATLossStreak5','HTWinStreak5','ATWinStreak5',
           'HTWinStreak3','HTLossStreak3','ATWinStreak3','ATLossStreak3'],1, inplace=True)

        # Separate into feature set and target variable
        X_all = data.drop(['FTR'],1)
        y_all = data['FTR']

        cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
        for col in cols:
            X_all[col] = scale(X_all[col])

        X_all.HM1 = X_all.HM1.astype('str')
        X_all.HM2 = X_all.HM2.astype('str')
        X_all.HM3 = X_all.HM3.astype('str')
        X_all.AM1 = X_all.AM1.astype('str')
        X_all.AM2 = X_all.AM2.astype('str')
        X_all.AM3 = X_all.AM3.astype('str')

        def preprocess_features(X):
            ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
            
            # Initialize new output DataFrame
            output = pd.DataFrame(index = X.index)

            # Investigate each feature column for the data
            for col, col_data in X.iteritems():

                # If data type is categorical, convert to dummy variables
                if col_data.dtype == object:
                    col_data = pd.get_dummies(col_data, prefix = col)
                            
                # Collect the revised columns
                output = output.join(col_data)
            
            return output

        X_all = preprocess_features(X_all)
        print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 50, random_state = 2, stratify = y_all)


        def predict_labels(clf, features, target):
            ''' Makes predictions using a fit classifier based on F1 score. '''
            
            # Start the clock, make predictions, then stop the clock
            start = time()
            print("-------------")
            print(type(features))
            print("---------------")
            y_pred = clf.predict(features)
            end = time()
            # Print and return results
            print("Made predictions in {:.4f} seconds.".format(end - start))
            
            return f1_score(target, y_pred, labels=['A','D','H'],average='micro'), sum(target == y_pred) / float(len(y_pred))

        # # TODO: Initialize the classifier
        f1_scorer = make_scorer(f1_score,labels=['A','D','H'],average='micro')
        parameters = { 'learning_rate' : [0.1],
                    'n_estimators' : [40],
                    'max_depth': [3],
                    'min_child_weight': [3],
                    'gamma':[0.4],
                    'subsample' : [0.8],
                    'colsample_bytree' : [0.8],
                    'scale_pos_weight' : [1],
                    'reg_alpha':[1e-5]
                    }  
        #clf.fit(X_train, y_train)
        logistic = LogisticRegression(random_state=42)
        svm = SVC(random_state=912, kernel='rbf')
        
        logistic.fit(X_train,y_train)
        f1, acc = predict_labels(logistic,X_test,y_test)
        print("Logistic Regression --> final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        svm.fit(X_train,y_train)
        f1, acc = predict_labels(svm,X_test,y_test)
        print("SVM --> final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        clf = xgb.XGBClassifier(seed=2)
        # # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
        grid_obj = GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)

        # # TODO: Fit the grid search object to the training data and find the optimal parameters
        grid_obj = grid_obj.fit(X_all,y_all)

        # # Get the estimator
        clf = grid_obj.best_estimator_
        #print(clf)

        # # Report the final F1 score for training and testing after parameter tuning
        f1, acc = predict_labels(clf, X_train, y_train)
        print("final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        f1, acc = predict_labels(clf, X_test, y_test)
        print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))
        data2 = pd.read_csv('team_dataframe.csv')
        data2 = data2.iloc[30:]
        global teamindex
        teamindex = 122
        for index, row in data2.iterrows():
            if teamname == row['HomeTeam']:
                teamindex = index
        #print(type(X_all.loc[x].to_frame().T))
        #print(X_all.loc[x].to_frame().T)
        winnerlist = clf.predict(X_all.loc[teamindex].to_frame().T)
        print(winnerlist)
        global teamwin
        global hnh
        teamwin = winnerlist[0]
        if teamwin == 'A':
            teamwin = team2
            hnh = "AwayTeam"
        elif teamwin == 'H':
            teamwin = team1
            hnh = "HomeTeam"
        else:
            teamwin = "DRAW!"
            hnh = "The game will be a DRAW"
        print(teamwin)
    else:
        return render_template('index.html', text='None',blehh='Please Select Different teams.')
    return render_template('index.html', text=teamwin,bleh=team2,blehh=hnh)



if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)
