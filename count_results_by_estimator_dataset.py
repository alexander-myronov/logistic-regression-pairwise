import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description='Results evaluation script')
parser.add_argument('csv', type=str, help='csv')
parser.add_argument('-p', '--percent', help='group by percents also', action="store_true")
args = parser.parse_args()

csv = pd.read_csv(args.csv)

print(csv.groupby(['dataset', 'estimator']).agg({'cv_split': 'count'}))
if args.percent:
    with pd.option_context('display.max_rows', None, 'display.max_columns', 7):
        print(
            csv.groupby(['dataset',
                         'estimator',
                         'percent_labels',
                         'percent_links',
                         'percent_unlabeled']).agg({'test_score': 'mean', 'cv_split': 'count'}))
