import pandas as pd
import numpy as np
import os

#Be sure to download the raw ANES cumulative data file in txt format from the ANES website
df = pd.read_csv(os.path.join("Data","anes_timeseries_cdf_rawdata.txt"))
mapping = {}
mapping["VCF0101"]  = "age"
mapping["VCF0104"]  = "gender"
mapping["VCF0105a"] = "race"
mapping["VCF0110"]  = "education"  
mapping["VCF0901b"] = "state"
mapping["VCF0004"]  = "survey year"
mapping["VCF0301"]  = "party"
mapping["VCF0803"]  = "ideology"
mapping["VCF0704a"] = "presVote"
mapping["VCF0009z"] = "weight"

presMap = {}
presMap[0] = "Other"
presMap[1] = "Democrat"
presMap[2] = "Republican"

genderMap = {}
genderMap[0] = "Other"
genderMap[1] = "Male"
genderMap[2] = "Female"
genderMap[3] = "Other"

raceMap = {}
raceMap[1] = "White non-Hispanic"
raceMap[2] = "Black non-Hispanic"
raceMap[3] = "Asian or PI non-Hispanic"
raceMap[4] = "American Indian or Alaska Native non-Hispanic"
raceMap[5] = "Hispanic"
raceMap[6] = "Other, non-Hispanic"
raceMap[7] = "Non-white and non-black"
raceMap[9] = "Missing"

educMap = {}
educMap[0] = "NA"
educMap[1] = "Noncollege"
educMap[2] = "Noncollege"
educMap[3] = "Noncollege"
educMap[4] = "College"

partyMap = {}
partyMap[0] = "Other"
partyMap[1] = "Democrat"
partyMap[2] = "Democrat"
partyMap[3] = "Democrat"
partyMap[4] = "Other"
partyMap[5] = "Republican"
partyMap[6] = "Republican"
partyMap[7] = "Republican"

ideoMap = {}
ideoMap[0] = "Other"
ideoMap[1] = "Democrat"
ideoMap[2] = "Democrat"
ideoMap[3] = "Democrat"
ideoMap[4] = "Other"
ideoMap[5] = "Republican"
ideoMap[6] = "Republican"
ideoMap[7] = "Republican"

df = df[mapping.keys()].rename(columns=mapping)
df["party"] = df["party"].replace({"":0})
df["party"] = df["party"].replace({" ":0})
df["party"] = df["party"].astype(int)
df["state"] = df["state"].astype(str)
df = df[df["state"]!="99"]
df = df[df["age"]!=0]
df = df[df["age"]!=""]
df["gender"] = df["gender"].replace(genderMap)
df["race"] = df["race"].replace(raceMap)
df["education"] = df["education"].replace(educMap)
df["presVote"] = df["presVote"].replace(presMap)
df["party"] = df["party"].replace(partyMap)
df["ideology"] = df["ideology"].replace(ideoMap)
df["age"] = df["age"].astype(float)
df = df[df["age"]!=0.]
df["survey year"] = df["survey year"].astype(float)
df["birthyear"] = df["survey year"] - df["age"]
df = df[df["birthyear"]>1930.]
print(df)
df.to_csv(os.path.join("Data","anes_cleaned.csv"),index=False)
