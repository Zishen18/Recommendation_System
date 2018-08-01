import math
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier

def train():
	
	trainDf = pd.read_csv("data_train.csv")
	X = np.mat(pd.DataFrame(trainDf, index=None, columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]))
	y = np.array(trainDf.interested)
	clf = SGDClassifier(loss="log", penalty="l2")
	clf.fit(X, y)
	return clf

def test(clf):
	
	origTestDf = pd.read_csv("../data/test.csv")
	users = origTestDf.user
	events = origTestDf.event
	testDf = pd.read_csv("data_test.csv")
	fout = open("result.csv", 'wb')
	fout.write(",".join(["user", "event", "outcome", "dist"]) + "\n")
	nrows = len(testDf)
	Xp = np.mat(testDf)
	yp = np.zeros((nrows, 2))
	for i in range(nrows):
		xp = Xp[i, :]
		yp[i, 0] = clf.predict(xp)
		yp[i, 1] = clf.decision_function(xp)
		fout.write(",".join(map(lambda x: str(x), [users[i], events[i], yp[i, 0], yp[i, 1]])) + "\n")
	fout.close()

clf = train()
test(clf)
