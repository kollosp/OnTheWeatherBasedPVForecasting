import pandas as pd
from os import listdir
from itertools import product
from os.path import isfile, join

mypath = "cm/aci"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)
df = pd.concat([pd.read_csv(mypath + "/" + path, header=[0,1],index_col=[0]) for path in onlyfiles])

ks = [2,3,4,5,6,7,8]
selectors = ["qACId"]
# selectors = ["qOCId", "qACId"]
selectors = [(f"{l[1]}({l[0]})", l[0], l[1]) for l in product(ks, selectors)]

df["selectors"] = ""
df["clearName"] = ""
df["k"] = 0

for index in df.index:
    for selector_name, k, selector in selectors:
        if selector_name in index:
            df.loc[index, "clearName"] = index.replace("," + selector_name, "")
            df.loc[index, "selectors"] = selector_name
            df.loc[index, "k"] = k



column = "FT" #"MAE"
# column = "MAE"

df = df[[(column, "Mean"), ("selectors", ""), ("k", "")]]
df.columns = df.columns.droplevel(1)
df["selectors"][df["selectors"] == ""] = "Ref."


uniques = df["selectors"].unique()
for u in uniques:
    u = u.replace("," + selector_name, "")
wide = pd.DataFrame({}, index=df[df["selectors"] == uniques[0]].index)
for m in uniques:
    print("m", m)
    c =  df[column][df["selectors"] == m].values
    print(c)
    wide[m] = c

wide = wide[["Ref."] + [c for c in wide.columns if c != "Ref."]]

print(wide)