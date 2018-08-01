import cPickle
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
from sklearn.preprocessing import normalize

def EventAttendees():
	
	eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
	nevents = len(eventIndex.keys())
	eventPopularity = ss.dok_matrix((nevents, 1))
	f = open("../data/event_attendees.csv", 'rb')
	f.readline()
	for line in f:
		cols = line.strip().split(",")
		eventId = cols[0]
		if eventIndex.has_key(eventId):
			i = eventIndex[eventId]
			eventPopularity[i, 0] = len(cols[1].split(" ")) - len(cols[4].split(" "))
	f.close()
	eventPopularity = normalize(eventPopularity, norm="l1", axis=0, copy=False)
	sio.mmwrite("EA_eventPopularity", eventPopularity)


