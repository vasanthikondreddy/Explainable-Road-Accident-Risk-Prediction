import pandas as pd, numpy as np


CSV = "AccidentsBig.csv"          # must be in the same folder as clean.py
OUT = "accidents_clean.csv"
d = pd.read_csv(CSV, on_bad_lines="skip")


print("raw lines:", len(d))
d = d.dropna(subset=["Accident_Index"]).drop_duplicates("Accident_Index")
print("non-blank rows:", len(d))

# 2. Date and time features
d = d.dropna(subset=["Time"])
d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
d["hour"] = pd.to_datetime(d["Time"], format="%H:%M").dt.hour
d["month"] = d["Date"].dt.month
d["is_weekend"] = d["Day_of_Week"].isin([1, 7]).astype(int)
d["time_of_day"] = pd.cut(d["hour"], [-1, 5, 11, 16, 20, 23],
                          labels=["Night", "Morning", "Afternoon", "Evening", "Late evening"])

# 3. -1 means unknown / not applicable
for c in d.columns:
    if d[c].dtype != object and c not in ("longitude", "latitude"):
        d[c] = d[c].replace(-1, np.nan)

# 4. Decode labels (standard STATS19 meanings, assumed)
maps = {
 "Accident_Severity": {1:"Fatal",2:"Serious",3:"Slight"},
 "Day_of_Week": {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"},
 "1st_Road_Class": {1:"Motorway",2:"A(M)",3:"A",4:"B",5:"C",6:"Unclassified"},
 "Road_Type": {1:"Roundabout",2:"One way street",3:"Dual carriageway",6:"Single carriageway",7:"Slip road",9:"Unknown",12:"One way/Slip road"},
 "Junction_Detail": {0:"Not at junction",1:"Roundabout",2:"Mini-roundabout",3:"T/staggered junction",5:"Slip road",6:"Crossroads",7:"More than 4 arms",8:"Private drive",9:"Other"},
 "Junction_Control": {0:"Not at junction",1:"Authorised person",2:"Traffic signal",3:"Stop sign",4:"Give way/uncontrolled"},
 "Light_Conditions": {1:"Daylight",4:"Dark - lit",5:"Dark - unlit",6:"Dark - no lighting",7:"Dark - unknown"},
 "Weather_Conditions": {1:"Fine",2:"Raining",3:"Snowing",4:"Fine + high winds",5:"Raining + high winds",6:"Snowing + high winds",7:"Fog/mist",8:"Other",9:"Unknown"},
 "Road_Surface_Conditions": {1:"Dry",2:"Wet/damp",3:"Snow",4:"Frost/ice",5:"Flood"},
 "Urban_or_Rural_Area": {1:"Urban",2:"Rural",3:"Unallocated"},
}
for col, m in maps.items():
    d[col + "_label"] = d[col].map(m).fillna("Unknown")

# 5. Keep only points inside India, drop UK admin codes
d = d[d["longitude"].between(68, 98) & d["latitude"].between(6, 37)]
d = d.drop(columns=["LSOA_of_Accident_Location", "Local_Authority_(District)",
                    "Local_Authority_(Highway)", "Police_Force"])
d["Accident_Severity"] = d["Accident_Severity"].astype(int)

d.to_csv(OUT, index=False)
print("clean rows:", len(d))
print(d["Accident_Severity_label"].value_counts().to_dict())