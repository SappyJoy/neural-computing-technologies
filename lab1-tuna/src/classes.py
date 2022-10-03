import pandas as pd

classes = pd.read_csv("../resources/labels/foid_labels_bbox_v012_freq.csv")
classes = classes.iloc[:, 1]
for classs in classes:
    print("'" + classs + "', ", end="")
