import datetime
import hashlib
import locale
import numpy as np
import pycountry
from collections import defaultdict

class Processor:
	
	
	# Get locale Id Map
	def GetlocaleIdMap(self):
		localeIdMap = defaultdict(int)
		i = 1
		for loc in locale.locale_alias.keys():
			localeIdMap[loc] = i
			i +=  1
		return localeIdMap
	
	#Get country ID map
	def GetcountryIdMap(self):
		countryIdMap = defaultdict(int)
		ctryId = defaultdict(int)
		i = 1
		for  country in pycountry.countries:
			countryIdMap[country.name.lower()] = i
			if country.name.lower() == "united states":
				ctryId["US"] = i
			if country.name.lower() == "canada":
				ctryId["CA"] = i
			i += 1
		for c in ctryId.keys():
			for s in pycountry.subdivisions.get(country_code=c):
				countryIdMap[s.name.lower()] = ctryId[c]
		return countryIdMap, ctryId			
	
	def GetgenderIdMap(self):
		return defaultdict(int, {"male":1, "female":2})
	
	def GetlocaleId(self, localeIdMap, locstr):
		return localeIdMap[locstr.lower()]

	def GetgenderId(self, genderIdMap, genderStr):
		return genderIdMap[genderStr]
	
	def GetJoinedYearMonth(self, dateString):
		dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
		return "".join([str(dttm.year), str(dttm.month)])	
	def GetCountryId(self, countryIdMap, location):
		if(isinstance(location, str) and len(location.strip()) > 0 and location.rfind("  ") > -1):
			return countryIdMap[location[location.rindex("  ") + 2:].lower()]
		else:
			return 0	

	def GetBirthYearInt(self, birthYear):
		try:
			return 0 if birthYear == "None" else int(birthYear)
		except:
			return 0
	def GetTimezoneInt(self, timezone):
		try:
			return int(timezone)
		except:
			return 0

	def GetFeatureHash(self, value):
		if len(value.strip()) == 0:
			return -1
		else:
			return int(hashlib.sha224(value).hexdigest()[0:4], 16)
	
	def GetFloatValue(self, value):
		if len(value.strip()) == 0:
			return 0.0
		else:
			return float(value)
			
