import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def train():


	trainDf = pd.read_csv("data_train.csv")
	testDf = pd.read_csv("data_test.csv")
	goal = "interested"
	predictors = ["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
		
	clf = XGBClassifier(learning_rate=0.1,
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
	
	X_train, X_test, y_train, y_test = train_test_split(trainDf[predictors], trainDf[goal], random_state=0)
	clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_test, y_test)])	
	return clf

def test(clf):
	testDf = pd.read_csv("data_test.csv")
	predictors = ["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]	
	prediction = clf.predict(testDf[predictors])
	origTestDf = pd.read_csv("../data/test.csv")
	users = origTestDf.user
	events = origTestDf.event
	proba = clf.predict_proba(testDf[predictors])
	print proba
	nrows = testDf.shape[0]
	fout = open("result.csv", 'wb')
	fout.write(",".join(["user", "event", "outcome", "prob"]) + "\n")
	for i in range(nrows):
		fout.write(",".join(map(lambda x: str(x), [users[i], events[i], prediction[i], proba[i][1]])) + "\n")
	fout.close()

clf = train()
test(clf)	
