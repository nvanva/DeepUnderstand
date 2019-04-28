import sys
import pickle
from evaluators import *

if len(sys.argv) < 5:
    print('Not enough arguments, at least 3 expected')
    print('eval_data.py [input_file] [data_type] [bert_checkpoint] [layer_count] [output_file]')
    sys.exit()
input_file = sys.argv[1]
data_type = sys.argv[2]
bert = sys.argv[3]

layers = '0'
for i in range(int(sys.argv[4]) - 1):
    layers += ',' + str(i+1)

if len(sys.argv) == 6:
    output_file = sys.argv[6]
else:
    output_file = data_type

if data_type == 'sentiment':
    metrics = eval_sentiment(input_file, bert, layers)
elif data_type == 'pos':
    metrics = eval_pos(input_file, bert, layers)
elif data_type == 'wsi':
    metrics = eval_wsi(input_file, bert, layers)
elif data_type == 'anaphora':
    metrics = eval_anaphora(input_file, bert, layers)
with open('outputs/' + output_file, 'wb') as f:
    pickle.dump(metrics, f)