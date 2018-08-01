import DataProcessor
import Entities
import EventAttendees
import Events
import UserFriends
import Users

def data_integration():
	
	print "Step 1: get user and event info..."
	uniqueUserPairs, uniqueEventPairs = Entities.Entities()
	print "Step 1 complete !"
	print "Step 2: compute user similarity matrix..."
	Users.Users(uniqueUserPairs)
	print "Step 2 completed !"
	print "Step 3: computer User social relationship..."
	UserFriends.UserFriends()
	print "Step 3 completed!"
	print "Step 4: computer event similarity matrix..."
	Events.Events(uniqueEventPairs)
	print "Step 4 completed!"
	print "Step 5: compute events popularity..."
	EventAttendees.EventAttendees()
	print "Step 5 completed!"	

data_integration()
