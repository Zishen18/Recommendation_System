import numpy as np
import scipy.sparse as ss
import scipy.io as sio
import cPickle
from sklearn.preprocessing import normalize

def UserFriends():
	
	userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
	userEventScores = sio.mmread("PE_userEventScores").todense()
	nusers = len(userIndex.keys())
	numFriends = np.zeros((nusers))
	userFriends = ss.dok_matrix((nusers, nusers))
	fin = open("../data/user_friends.csv", 'rb')
	fin.readline()
	ln = 0
	for line in fin:
		if ln % 200 == 0:
			print "Loading line: ", ln
		cols = line.strip().split(",")
		user = cols[0]
		if userIndex.has_key(user):
			friends = cols[1].split(" ")
			i = userIndex[user]
			numFriends[i] = len(friends)
			for friend in friends:
				if userIndex.has_key(friend):
					j = userIndex[friend]
					eventsForUser = userEventScores[j]
					score = eventsForUser.sum() / np.shape(eventsForUser)[1]
					userFriends[i, j] += score
					userFriends[j, i] += score
		ln += 1
	fin.close()
	sumNumFriends = numFriends.sum(axis=0)
	numFriends = numFriends / sumNumFriends
	sio.mmwrite("UF_numFriends", np.mat(numFriends))
	userFriends = normalize(userFriends, norm="l1", axis=0, copy=False)
	sio.mmwrite("UF_userFriends", userFriends)


