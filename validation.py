import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier

def validate():
	
	trainDf = pd.read_csv("data_train.csv")
	X = np.matrix(pd.DataFrame(trainDf, index=None, columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]))
	y = np.array(trainDf.interested)
	nrows = len(trainDf)
	kfold = KFold(nrows, 10)
	avgAccuracy = 0
	run = 0
	for train, test in kfold:
		Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
		clf = SGDClassifier(loss="log", penalty="l2")
		clf.fit(Xtrain, ytrain)
		accuracy = 0
		ntest = len(ytest)
		for i in range(ntest):
			yt = clf.predict(Xtest[i, :])
			if yt == ytest[i]:
				accuracy += 1
		accuracy = accuracy / ntest
		print "accuracy (run %d): %f" %(run, accuracy)
		avgAccuracy += accuracy
		run += 1
	print "Average accuracy", (avgAccuracy / run)

validate()
