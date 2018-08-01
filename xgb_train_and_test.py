import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
def train():
	trainDf = pd.read_csv("data_train.csv")
        X = np.mat(pd.DataFrame(trainDf, index=None, columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]))
        y = np.array(trainDf.interested)
	params = {"booster": "gbtree",
		  "objective": "binary:logistic",
		  "eval_metric": "error",
		  "eta": 0.3,
		  "gamma": 0,
		  "max_depth": 6,
		  "min_child_weight": 1,
		  "max_delta_step": 0,
		  "subsample": 1,
		  "colsample_bytree": 1,
		  "silent": 1,
		  "seed": 0,
		  "base_score": 0.5
		  }
	
	clf = xgb.XGBClassifier(params)
	metLearn = CalibratedClassifierCV(clf, method='isotonic', cv=2)
	metLearn.fit(X, y)
	return metLearn

def test(metLearn):
	
	origTestDf = pd.read_csv("../data/test.csv")
        users = origTestDf.user
        events = origTestDf.event
        testDf = pd.read_csv("data_test.csv")
        fout = open("xgb_result.csv", 'wb')
        fout.write(",".join(["user", "event", "outcome"]) + "\n")
        nrows = len(testDf)
        Xp = np.mat(testDf)
        yp = np.zeros((nrows, 1))
        for i in range(nrows):
                xp = Xp[i, :]
                yp[i, 0] = metLearn.predict(xp)
                fout.write(",".join(map(lambda x: str(x), [users[i], events[i], yp[i, 0]])) + "\n")
        fout.close()

metLearn = train()
test(metLearn)
