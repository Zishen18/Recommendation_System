import numpy as np
import scipy.io as sio

def userReco(userIndex, eventIndex, userEventScores, userSimMatrix, userId, eventId):
	
	i = userIndex[userId]
	j = eventIndex[eventId]
	event_score = userEventScores[:, j]
	user_sim = userSimMatrix[i, :]
	prod = user_sim * event_score
	
	return prod[0, 0] - userEventScores[i, j]


def eventReco(userIndex, eventIndex, userEventScores, eventPropSim, eventContSim, userId, eventId):
	
	i = userIndex[userId]
	j = eventIndex[eventId]
	event_score = userEventScores[i, :]
	psim = eventPropSim[:, j]
	csim = eventContSim[:, j]
	pprod = event_score * psim
	cprod = event_score * csim
	pscore = pprod[0, 0] - userEventScores[i, j]
	cscore = cprod[0, 0] - userEventScores[i, j]
	return pscore, cscore

def userPop(userIndex, numFriends, userId):
	
	if userIndex.has_key(userId):
		i = userIndex[userId]
		return numFriends[0, i]
	else:
		return 0

def friendInfluence(userId, userIndex, userFriends):

	nusers = np.shape(userFriends)[0]
	i = userIndex[userId]
	return (userFriends[i, :].sum(axis=1) / nusers)[0, 0]

def eventPop(eventPopularity, eventIndex, eventId):
	
	i = eventIndex[eventId]
	return eventPopularity[i, 0]


