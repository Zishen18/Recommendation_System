import pandas as pd

def byDist(x, y):
	return int(y[1] - x[1])

def generate_submission_file():
		
	fout = open("final_result.csv", 'wb')
	fout.write(",".join(["User", "Events"]) + "\n")
	
	resultDf = pd.read_csv("result.csv")
	grouped = resultDf.groupby("user")
	for name, group in grouped:
		user = str(name)
		tuples = zip(list(group.event), list(group.prob), list(group.outcome))
		tuples = sorted(tuples, cmp=byDist)
		events = "\"" + str(map(lambda x: x[0], tuples)) + "\""
		fout.write(",".join([user, events]) + "\n")
	fout.close()

generate_submission_file()
