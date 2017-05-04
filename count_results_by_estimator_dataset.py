import pandas as pd
import sys
csv = pd.read_csv(sys.argv[1])
print(csv.groupby(['dataset', 'estimator']).agg({'cv_split':'count'}))
