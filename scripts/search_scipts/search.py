import json
import os
import sys

from glob import glob

from tasks.utils import *

TASK = sys.argv[1]
MODEL = sys.argv[2]
METHOD = sys.argv[3]


SPECIAL_METRICS = {
    'cb' : 'f1',
    'mrpc' : 'f1',
    'cola' : 'matthews_correlation',
    'stsb' : 'combined_score'
}

METRIC = "accuracy"
if TASK in SPECIAL_METRICS:
    METRIC = SPECIAL_METRICS[TASK]

best_score = 0

files = glob(f"./checkpoints/{TASK}-{MODEL}-search{METHOD}/*/predict_results.json")

for f in files:
    metrics = json.load(open(f, 'r'))
    if metrics["predict_"+METRIC] > best_score:
        best_score = metrics["predict_"+METRIC]
        best_metrics = metrics
        best_file_name = f

print(f"best_{METRIC}: {best_score}")
print(f"best_metrics: {best_metrics}")
print(f"best_file: {best_file_name}")
