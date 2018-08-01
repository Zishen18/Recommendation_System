import cPickle
import datetime
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
import itertools
from collections import defaultdict

def Entities():
	
	UserSet = set()
	EventSet = set()
	eventsForUser = defaultdict(set)
	usersForEvent = defaultdict(set)
	
	for filename in ["../data/train.csv", "../data/test.csv"]:
		file = open(filename, 'rb')
		file.readline()
		for line in file:
			cols = line.strip().split(",")
			UserSet.add(cols[0])
			EventSet.add(cols[1])
			eventsForUser[cols[0]].add(cols[1])
			usersForEvent[cols[1]].add(cols[0])
		file.close()
	
	userEventScores = ss.dok_matrix((len(UserSet), len(EventSet)))
	userIndex = dict()
	eventIndex = dict()
	
	for i, u in enumerate(UserSet):
		userIndex[u] = i
	for i, e in enumerate(EventSet):
		eventIndex[e] = i
	
	ftrain = open("../data/train.csv", 'rb')
	ftrain.readline()
	for line in ftrain:
		cols = line.strip().split(",")
		i = userIndex[cols[0]]
		j = eventIndex[cols[1]]
		userEventScores[i, j] = int(cols[4]) - int(cols[5])
	ftrain.close()
	sio.mmwrite("PE_userEventScores", userEventScores)
	
	#unique pairs
	uniqueUserPairs = set()
	uniqueEventPairs = set()
	
	for event in EventSet:
		users = usersForEvent[event]
		if len(users) > 2:
			uniqueUserPairs.update(itertools.combinations(users, 2))
	
	for user in UserSet:
		events = eventsForUser[user]
		if len(events) > 2:
			uniqueEventPairs.update(itertools.combinations(events, 2))
	cPickle.dump(userIndex, open("PE_userIndex.pkl", 'wb'))
	cPickle.dump(eventIndex, open("PE_eventIndex.pkl", 'wb'))
	return uniqueUserPairs, uniqueEventPairs

