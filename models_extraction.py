from sklearn import svm
import visualization
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb


# Shuffle and split the dataset into training and testing set.
X_all = visualization.X_all
y_all = visualization.y_all
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = y_all)

# Fitting Logistic Regression to the Training set
regression = LogisticRegression(random_state = 0)
regression.fit(X_train, y_train)


# #Comment out to Show confusion matrix of Logistic Regression
# Y_pred = regression.predict(X_test)
# cm_regression = confusion_matrix(y_test, Y_pred)
# print(classification_report(y_test, Y_pred))
# sns.heatmap(cm_regression, annot=True,fmt='d')
# plt.show(block=True)

#Fitting the SVM to the training set
svm_model = SVC(kernel = 'rbf',random_state = 0)
svm_model.fit(X_train, y_train)

# #Comment out to Show confusion matrix of SVM
# Y_pred = svm_model.predict(X_test)
# cm_svm = confusion_matrix(y_test, Y_pred)
# sns.heatmap(cm_svm, annot=True, fmt='d')
# plt.show(block=True)


#Fitting XGBoost to the Training set
xgboostmodel = XGBClassifier(seed=82)
xgboostmodel.fit(X_train, y_train)

# #Comment out to Show confusion matrix of XGBoost
# Y_pred = xgboostmodel.predict(X_test)
# cm_xgboost = confusion_matrix(y_test, Y_pred)
# sns.heatmap(cm_xgboost, annot=True,fmt='d')
# plt.show(block=True)

#Tuning of XGBoost
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

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    y_pred = clf.predict(features)
    
    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


# TODO: Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score,pos_label='H')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print( "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))




#Getting models outside by pickle
pickle.dump(regression, open('regression_model','wb'))
pickle.dump(svm_model, open('svm_model','wb'))
pickle.dump(xgboostmodel, open('xgboostmodel','wb'))
pickle.dump(clf, open('tunedmodel','wb'))

