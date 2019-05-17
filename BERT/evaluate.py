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

if len(sys.argv) >= 6:
    output_file = sys.argv[5]
else:
    output_file = data_type

if data_type == 'anaphora':
    output_file = 'anaphora_' + output_file

if data_type == 'sentiment':
    metrics, fold_count, data_size, random_sample, class_sizes = eval_sentiment(input_file, bert, layers)
    rows = [['class', 'text']]
    # for sample in random_sample:
    #     rows.append(sample)
    rows += random_sample
elif data_type == 'sentiment_b':
    metrics, fold_count, data_size, random_sample, class_sizes = eval_sentiment_big(input_file, bert, layers)
    rows = [['class', 'text']]
    # for sample in random_sample:
    #     rows.append(sample)
    rows += random_sample
elif data_type == 'pos':
    metrics, fold_count, data_size, random_sample, class_sizes = eval_pos(input_file, bert, layers)
    rows = [['word'], ['tag']]
    for sample in random_sample:
        for word in sample:
            rows[0].append(word[0])
            rows[1].append(word[1])
elif data_type == 'wsi':
    metrics, fold_count, data_size, random_sample, class_sizes = eval_wsi(input_file, bert, layers)
    rows = [['text', 'target idx', 'class']]
    rows += random_sample
elif data_type == 'anaphora':
    metrics = eval_anaphora(input_file, bert, layers)
    with open('outputs/' + output_file, 'wb') as f:
        pickle.dump(metrics, f)


with open('outputs/' + output_file, 'wb') as f:
    pickle.dump((fold_count, data_size, metrics, rows, class_sizes), f)