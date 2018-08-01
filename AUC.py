# Reference: http://blog.csdn.net/u010454729/article/details/45098305

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold


def plotAUC():
	
	trainDf = pd.read_csv("data_train.csv")
	goal = "interested"
	predictors = ["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
	X = trainDf[predictors].as_matrix()
	y = trainDf[goal].as_matrix()
	n_samples, n_features = X.shape

	cv = StratifiedKFold(y, n_folds=6)
	# original parameter
	'''
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
	'''
	clf = XGBClassifier(base_score=0.5,
			    booster='gbtree',
			    colsample_bylevel=1,
			    colsample_bytree=0.8,
			    gamma=0,
			    learning_rate=0.1,
			    max_delta_step=0,
			    max_depth=5,
			    min_child_weight=1,
			    missing=None,
			    n_estimators=685,
			    n_jobs=4,
			    nthread=4,
			    objective='binary:logistic',
			    random_state=27,
			    reg_alpha=0,
			    reg_lambda=1,
			    scale_pos_weight=1,
			    seed=27,
			    silent=True,
			    subsample=0.8)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []
	
	for i, (train, test) in enumerate(cv):
		probas = clf.fit(X[train], y[train]).predict_proba(X[test])
		
		fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
	
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='ideal')
	
	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	
	plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

plotAUC()
