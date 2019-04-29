from typing import TextIO

import pandas as pd

df = pd.read_csv("state_entries.csv")
print(df.keys())
df = df["State"].values
file_obj = open("state.txt", 'w')
for elem in df:
    file_obj.write("\"{}\",".format(elem))
file_obj.close()

