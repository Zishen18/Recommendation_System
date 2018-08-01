import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	target = 'interested'	
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
		xgtest = xgb.DMatrix(dtest[predictors].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
		alg.set_params(n_estimators=cvresult.shape[0])
	
	# modeling
	alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
	
	# make prediction
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
	
	print "about this model: "
	print "accuracy: %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
	auc = metrics.roc_auc_score(dtrain[target], dtrain_predprob)
	print "AUC score (training set): %f" % auc
	return auc
	
	
	'''		
	feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
	plt.show()
	'''
	
	

def find_best_estimators():
	predictors = [x for x in test.columns]	
	xgb1 = XGBClassifier(learning_rate=0.1,
                             n_estimators=1000,
                             max_depth=5,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective='binary:logistic',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=27)
		
	auc = modelfit(xgb1, train, test, predictors)
	print "n_estimators updated!"
	return xgb1, auc

def parameter_tuning(xgbm, param_test):
	predictors = [x for x in test.columns]
	target = 'interested'
	print "Grid Search Cross Validation for parameters: "
	print param_test
	gsearch = GridSearchCV(estimator=xgbm, param_grid=param_test, scoring='roc_auc', verbose=2, iid=False, cv=5)
	print "Fitting data..."
	DTrain = train[predictors].as_matrix()
	Dy = pd.Series(train[target])
	gsearch.fit(DTrain, Dy)
	print "Complete fitting..."
	print "grid_scores_"
	print gsearch.grid_scores_
	print "Grid Search best_params_"
	print gsearch.best_params_
	print "Best score"
	print gsearch.best_score_
	print "Complete!"
	return gsearch.estimator, gsearch.best_params_, gsearch.best_score_

def update_params():
	
	print "Find the best estimators..."
	
	xgb1, auc = find_best_estimators()
	max_auc = auc
	best_model = xgb1
	print "model xgb1: "
	print xgb1

	# Tune max_depth and min_child_weight
	print "Grid Search: max_depth, min_child_weight"
	param_test = {
        	'max_depth': range(4,10,2),
                'min_child_weight': range(1,6,2)
        }
	xgb2, best_params_, best_score_2 = parameter_tuning(xgb1, param_test)
	if(best_score_2 > max_auc):
		max_auc = best_score_2
		best_model.set_params(max_depth=best_params_['max_depth'])
		best_model.set_params(min_child_weight=best_params_['min_child_weight'])
		print "best model: "
		print best_model
		print "best auc = ", max_auc

	# Tune gamma
	print "Grid Search: max_depth, min_child_weight"
        param_test = {
                'gamma': [i/10.0 for i in range(0, 5)]
                
        }
        xgb3, best_params_, best_score_3 = parameter_tuning(xgb2, param_test)
	if(best_score_3 > max_auc):
		max_auc = best_score_3
        	best_model.set_params(gamma=best_params_['gamma'])
        
        	print "best model: "
        	print best_model
		print "best auc = ", max_auc

	# Tune subsample and colsample_bytree
	print "Grid Search: subsample, colsample_bytree"
	param_test = {
                'subsample': [i/10.0 for i in range(6, 10)],
		'colsample_bytree': [i/10.0 for i in range(6, 10)]

        }
        xgb4, best_params_, best_score_4 = parameter_tuning(xgb3, param_test)
	if(best_score_4 > max_auc):
		max_auc = best_score_4
        	best_model.set_params(subsample=best_params_['subsample'])
		best_model.set_params(colsample_bytree=best_params_['colsample_bytree'])

        	print "best model: "
        	print best_model
		print "best auc = ", max_auc
	
	# Tune learning rate
	print "Grid Search: subsample, colsample_bytree"
        param_test = {
                'learning_rate': [0.05, 0.01],
                'n_estimators': [1370, 6850]

        }
        xgb5, best_params_, best_score_5 = parameter_tuning(xgb4, param_test)
        if(best_score_5 > max_auc):
		max_auc = best_score_5
                best_model.set_params(learning_rate=best_params_['learning_rate'])
                best_model.set_params(n_estimators=best_params_['n_estimators'])

                print "best model: "
                print best_model
		print "best auc = ", max_auc
	
	print "Final Result: "
	print "Best Model: "
	print best_model
	print "best auc = ", max_auc			
update_params()

