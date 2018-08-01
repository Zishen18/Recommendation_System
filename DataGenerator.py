import Create_Feature
import cPickle
import scipy.io as sio

def Generator(train=True, header=True):
	print "loading data ..."
	userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
	eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
	userEventScores = sio.mmread("PE_userEventScores").todense()
	userSimMatrix = sio.mmread("US_userSimMatrix").todense()
	eventPropSim = sio.mmread("EV_eventPropSim").todense()
	eventContSim = sio.mmread("EV_eventContSim").todense()
	numFriends = sio.mmread("UF_numFriends")
	userFriends = sio.mmread("UF_userFriends").todense()
	eventPopularity = sio.mmread("EA_eventPopularity").todense()
	
	file = "train.csv" if train else "test.csv"
	fin = open("../data/" + file, 'rb')
	fout = open("data_" + file, 'wb')
	
	if header:
		colnames = ["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
		if train:
			colnames.append("interested")
			colnames.append("not_interested")
		fout.write(",".join(colnames) + "\n")
	ln = 0
	fin.readline() # skip header
	for line in fin:
		cols = line.strip().split(",")
		userId = cols[0]
		eventId = cols[1]
		invited = cols[2]
		if ln % 500 == 0:
			print "%s: %d (userId, eventId)=(%s, %s)" % (file, ln, userId, eventId)
		user_reco = Create_Feature.userReco(userIndex, eventIndex, userEventScores, userSimMatrix, userId, eventId)
		evt_p_reco, evt_c_reco = Create_Feature.eventReco(userIndex, eventIndex, userEventScores, eventPropSim, eventContSim, userId, eventId)
		user_pop = Create_Feature.userPop(userIndex, numFriends, userId)
		frnd_infl = Create_Feature.friendInfluence(userId, userIndex, userFriends)
		evt_pop = Create_Feature.eventPop(eventPopularity, eventIndex, eventId)
		col_val = [invited, user_reco, evt_p_reco, evt_c_reco, user_pop, frnd_infl, evt_pop]
		if train:
			col_val.append(cols[4])
			col_val.append(cols[5])
		fout.write(",".join(map(lambda x: str(x), col_val)) + "\n")
		ln += 1
	fin.close()
	fout.close()

print "Generating training data..."
Generator(train=True, header=True)
print "Generating test data..."
Generator(train=False, header=True)

		
