import cPickle
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import Entities
import DataProcessor
from sklearn.preprocessing import normalize
import scipy.spatial.distance as ssd
import Entities
import DataProcessor

def Events(uniqueEventPairs):
	
	#uniqueUserPairs, uniqueEventPairs = Entities.Entities()	
	processor = DataProcessor.Processor()
	eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
	fin = open("../data/events.csv", 'rb')
	fin.readline()
	nevents = len(eventIndex.keys())
	eventPropMatrix = ss.dok_matrix((nevents, 7))
	eventContMatrix = ss.dok_matrix((nevents, 100))
	ln = 0
	for line in fin.readlines():
		cols = line.strip().split(",")
		eventId = cols[0]
		if eventIndex.has_key(eventId):
			i = eventIndex[eventId]
			eventPropMatrix[i, 0] = processor.GetJoinedYearMonth(cols[2])
			eventPropMatrix[i, 1] = processor.GetFeatureHash(cols[3])
			eventPropMatrix[i, 2] = processor.GetFeatureHash(cols[4])
			eventPropMatrix[i, 3] = processor.GetFeatureHash(cols[5])
			eventPropMatrix[i, 4] = processor.GetFeatureHash(cols[6])
			eventPropMatrix[i, 5] = processor.GetFloatValue(cols[7])
			eventPropMatrix[i, 6] = processor.GetFloatValue(cols[8])
			for j in range(9, 109):
				eventContMatrix[i, j-9] = cols[j]
			ln += 1
	fin.close()
	
	eventPropMatrix = normalize(eventPropMatrix, norm="l1", axis=0, copy=False)
	sio.mmwrite("EV_eventProMatrix", eventPropMatrix)
	eventContMatrix = normalize(eventContMatrix, norm="l1", axis=0, copy=False)
	sio.mmwrite("EV_eventContMatrix", eventContMatrix)
	eventPropSim = ss.dok_matrix((nevents, nevents))
	eventContSim = ss.dok_matrix((nevents, nevents))
	for e1, e2 in uniqueEventPairs:
		i = eventIndex[e1]
		j = eventIndex[e2]
		if not eventPropSim.has_key((i, j)):
			epsim = ssd.correlation(eventPropMatrix.getrow(i).todense(), eventPropMatrix.getrow(j).todense())
			eventPropSim[i, j] = epsim
			eventPropSim[j, i] = epsim
		if not eventContSim.has_key((i, j)):
			ecsim = ssd.cosine(eventContMatrix.getrow(i).todense(), eventContMatrix.getrow(j).todense())
			eventContSim[i, j] = epsim
			eventContSim[j, i] = epsim
	sio.mmwrite("EV_eventPropSim", eventPropSim)	
	sio.mmwrite("EV_eventContSim", eventContSim)



