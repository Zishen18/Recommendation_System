from __future__ import division
import cPickle
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import Entities
import DataProcessor
from sklearn.preprocessing import normalize
import scipy.spatial.distance as ssd

def Users(uniqueUserPairs):
	
	processor = DataProcessor.Processor()
	localeIdMap = processor.GetlocaleIdMap()
	genderIdMap = processor.GetgenderIdMap()
	countryIdMap, ctryId = processor.GetcountryIdMap()
	print "Country Id Map..."
	print countryIdMap
	
	userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
	#eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
	#userEventScores = sio.mmread("PE_userEventScores").todense()
	nusers = len(userIndex.keys())
	fin = open("../data/users.csv", 'rb')
	colnames = fin.readline().strip().split(",")
	userMatrix = ss.dok_matrix((nusers, len(colnames) - 1))
	
	for line in fin:
		cols = line.strip().split(",")
		if userIndex.has_key(cols[0]):
			i = userIndex[cols[0]]
			userMatrix[i, 0] = processor.GetlocaleId(localeIdMap, cols[1])
			userMatrix[i, 1] = processor.GetBirthYearInt(cols[2])
			userMatrix[i, 2] = processor.GetlocaleId(genderIdMap, cols[3])
			userMatrix[i, 3] = processor.GetJoinedYearMonth(cols[4])
			userMatrix[i, 4] = processor.GetCountryId(countryIdMap, cols[5])
			userMatrix[i, 5] = processor.GetTimezoneInt(cols[6])
	fin.close()
	print "user Matrix: "
	print userMatrix.todense()
	
	# normalize user matrix
	userMatrix = normalize(userMatrix, norm="l1", axis=0, copy=False)
	sio.mmwrite("US_userMatrix", userMatrix)
	
	# compute user sim matrix
	userSimMatrix = ss.dok_matrix((nusers, nusers))
	for i in range(nusers):
		userSimMatrix[i, i] = 1.0

	for u1, u2 in uniqueUserPairs:
		i = userIndex[u1]
		j = userIndex[u2]
		
		if not userSimMatrix.has_key((i, j)):
			usim = ssd.correlation(userMatrix.getrow(i).todense(), userMatrix.getrow(j).todense())
			userSimMatrix[i, j] = usim
			userSimMatrix[j, i] = usim
	sio.mmwrite("US_userSimMatrix", userSimMatrix)

	
