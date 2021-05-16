from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

 
 
folder = Path("evaluation/spreadsheets/charuco_CH1_35-15")
csv_files = list(folder.glob("*.csv"))

dataframes = detected = {}
for file in csv_files:
    dataframes[file.stem] = pd.read_csv(file)

    detected[file.stem] = dataframes[file.stem]["e_ids"] + dataframes[file.stem]["ids"]


    # print(dataframes[file.stem][["ids", "e_ids"]])

# dataframes["1to1"][["ids", "e_ids"]].plot.line()
# df = dataframes["1to1"]["e_ids"] + dataframes["1to1"]["ids"]
detected["1to1"].plot.line()

plt.show()


 