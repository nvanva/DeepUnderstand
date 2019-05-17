import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
from bert.tokenization import FullTokenizer
sys.path.insert(0, 'supervisedPCA-Python')
from supervised_pca import SupervisedPCARegressor
from metrics import *
import csv

def eval_sentiment_big(input_file, bert, layers, run=True, batch_size=400):
    labels = []
    line_count = 0
    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
    random_sample_idx = np.random.choice(line_count, 2)
    random_sample = []

    with open(input_file, 'r') as f:
        with open('texts', 'w') as dest:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i in random_sample_idx:
                    random_sample.append(row)
                # if i == 52:
                #     break
                labels.append(int(row[0]))
                dest.write(row[1] + '\n')


    if run:
        args = ['python', 'bert/extract_features.py']
        args.append('--input_file=texts')
        args.append('--output_file=sentiment')
        args.append('--vocab_file=' + bert + 'vocab.txt')
        args.append('--bert_config_file=' + bert + 'bert_config.json')
        args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
        args.append('--layers=' + layers)
        args.append('--max_seq_length=128')
        args.append('--batch_size=8')
        args.append('--do_lower_case=False')
        args.append('--attention=False')
        args.append('--mask_underscore=False')
        subprocess.run(args)

    layer_cls_metrics = []
    layer_v_metrics = []
    layer_avg_dist_metrics = []
    for layer in range(len(layers.split(','))):
        outputs = []
        for batch in range(1000):
            try:
                layer_outputs = np.load('layer_outputs/sentiment_layer_' + str(layer) + '_' + str(batch) + '.npz')
                try:
                    outputs += [layer_outputs['arr_' + str(x)][0] for x in range(batch_size)]
                except:
                    outputs += [layer_outputs['arr_' + str(x)][0] for x in range(len(labels) % batch_size)]
            except:
                break
        
        outputs = np.array(outputs)
        pca_outputs = outputs

        class_vectors = [[], []]
        for idx, label in enumerate(labels):
            class_vectors[label].append(pca_outputs[idx])

        layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1, validate=True))
        layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
        layer_avg_dist_metrics.append(distance_metrics(class_vectors))


    metrics = []
    for idx, cls_metric in enumerate(layer_cls_metrics):
        metrics.append((f'Layer {idx + 1} : avg_clas_score {cls_metric[0]}, min_val_score {cls_metric[1]}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}, conf_interval {cls_metric[3]}', cls_metric[0], cls_metric[1], layer_v_metrics[idx], layer_avg_dist_metrics[idx], cls_metric[5], cls_metric[7], cls_metric[6], cls_metric[2], cls_metric[3], cls_metric[4], idx))
    unique, counts = np.unique(labels, return_counts=True)
    return metrics, 15, len(labels), random_sample, list(zip(unique, counts))


def eval_sentiment(input_file, bert, layers, run=True):
    labels = []

    line_count = 0
    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
    random_sample_idx = np.random.choice(line_count, 2)
    random_sample = []

    with open(input_file, 'r') as f:
        with open('texts', 'w') as dest:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i in random_sample_idx:
                    random_sample.append(row)
                # if i == 52:
                #     break
                labels.append(int(row[0]))
                dest.write(row[1] + '\n')


    if run:
        args = ['python', 'bert/extract_features.py']
        args.append('--input_file=texts')
        args.append('--output_file=sentiment')
        args.append('--vocab_file=' + bert + 'vocab.txt')
        args.append('--bert_config_file=' + bert + 'bert_config.json')
        args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
        args.append('--layers=' + layers)
        args.append('--max_seq_length=128')
        args.append('--batch_size=8')
        args.append('--do_lower_case=False')
        args.append('--attention=False')
        args.append('--mask_underscore=False')
        subprocess.run(args)




    layer_cls_metrics = []
    layer_v_metrics = []
    layer_avg_dist_metrics = []
    for layer in range(len(layers.split(','))):
        layer_outputs = np.load('sentiment_layer_' + str(layer) + '.npz')
        outputs = np.array([layer_outputs['arr_' + str(x)][0] for x in range(len(labels))]) # arrays of shape (num_texts, num_neurons) 


        try:
            pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
            pca = SupervisedPCARegressor(threshold = 0.8)
            # pca = SupervisedPCARegressor(threshold=0.1)
            pca.fit(outputs, labels)

            pca_outputs = pca.get_transformed_data(outputs)
            print(pca_outputs.shape)
        except:
            pca_outputs = outputs

        class_vectors = [[], []]
        for idx, label in enumerate(labels):
            class_vectors[label].append(pca_outputs[idx])

        layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1, validate=True))
        layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
        layer_avg_dist_metrics.append(distance_metrics(class_vectors))


    metrics = []
    for idx, cls_metric in enumerate(layer_cls_metrics):
        metrics.append((f'Layer {idx + 1} : avg_clas_score {cls_metric[0]}, min_val_score {cls_metric[1]}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}, conf_interval {cls_metric[3]}', cls_metric[0], cls_metric[1], layer_v_metrics[idx], layer_avg_dist_metrics[idx], cls_metric[5], cls_metric[7], cls_metric[6], cls_metric[2], cls_metric[3], cls_metric[4], idx))
    unique, counts = np.unique(labels, return_counts=True)
    return metrics, 15, len(labels), random_sample, list(zip(unique, counts))

def eval_pos(input_file, bert, layers, run=True):
    data = np.load(input_file, allow_pickle=True)
    with open('sentences', 'w') as f:
        for text in data:
                string = ''
                for word in text:
                        string += word[0] + ' '
                f.write(string[:-1] + '\n')

    bert_tokens = []
    token_map = []
    tokenizer = FullTokenizer(vocab_file=bert + 'vocab.txt', do_lower_case=False)

    for text in data:
        text_tokens = ['[CLS]']
        text_map = []
        for word in text:
            text_map.append(len(text_tokens))
            text_tokens.extend(tokenizer.tokenize(word[0]))

        token_map.append(text_map)
        bert_tokens.append(text_tokens)

    if run:
        args = ['python', 'bert/extract_features.py']
        args.append('--input_file=sentences')
        args.append('--output_file=POS')
        args.append('--vocab_file=' + bert + 'vocab.txt')
        args.append('--bert_config_file=' + bert + 'bert_config.json')
        args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
        args.append('--layers=' + layers)
        args.append('--max_seq_length=128')
        args.append('--batch_size=8')
        args.append('--do_lower_case=False')
        args.append('--attention=False')
        args.append('--mask_underscore=False')
        subprocess.run(args)


    pos_to_idx = {}
    idx_to_pos = {}
    max_label = 0
    correct = []
    for text in data:
        labels = []
        for word in text:
            if word[1] not in pos_to_idx:
                pos_to_idx[word[1]] = max_label
                idx_to_pos[max_label] = word[1]
                max_label += 1
            labels.append(pos_to_idx[word[1]])
        correct.append(labels)


    layer_cls_metrics = []
    layer_v_metrics = []
    layer_avg_dist_metrics = []
    for layer in range(len(layers.split(','))):
        layer_outputs = np.load('POS_layer_' + str(layer) + '.npz')
        text_outputs_raw = [layer_outputs['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_tokens, num_neurons) 

        # remove vectors from excess tokens
        text_outputs = []
        for idx, text_output in enumerate(text_outputs_raw):
            text_outputs.append(text_output[token_map[idx]])



        class_vectors = [[] for _ in range(max_label)]
        for t_idx, text in enumerate(data):
            for w_idx, word in enumerate(text):
                # do something with multiple tokens per word, for example all tokens are the same pos
                class_vectors[pos_to_idx[word[1]]].append(text_outputs[t_idx][w_idx])

        labels = []
        for idx, vectors in enumerate(class_vectors):
            labels += [idx for _ in range(len(vectors))]

        outputs = np.concatenate(class_vectors, axis=0)


        try:
            pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
            pca = SupervisedPCARegressor(threshold = 0.7)
            # pca = SupervisedPCARegressor(threshold=0.1)
            pca.fit(outputs, labels)

            pca_outputs = pca.get_transformed_data(outputs)
            print(pca_outputs.shape)
        except:
            pca_outputs = outputs


        layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1, validate=True, fold_count=10))
        layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
        layer_avg_dist_metrics.append(distance_metrics(class_vectors))   


    metrics = []
    for idx, cls_metric in enumerate(layer_cls_metrics):
        metrics.append((f'Layer {idx + 1} : avg_clas_score {cls_metric[0]}, min_val_score {cls_metric[1]}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}, conf_interval {cls_metric[3]}', cls_metric[0], cls_metric[1], layer_v_metrics[idx], layer_avg_dist_metrics[idx], cls_metric[5], cls_metric[7], cls_metric[6], cls_metric[2], cls_metric[3], cls_metric[4], idx))
    
    unique1, counts = np.unique(np.concatenate(correct), return_counts=True)
    unique = []
    for x in unique1:
        unique.append(idx_to_pos[x])
    return metrics, 10, np.sum(list(map(len, data))), np.random.choice(data, 1), list(zip(unique, counts))

def eval_wsi(input_file, bert, layers, run=True):
    data = np.load(input_file, allow_pickle=True)#[:10]
    data = [(ex[0], int(ex[1]), int(ex[2])) for ex in data]

    with open('texts', 'w') as f:
        for example in data:
            f.write(example[0] + '\n')


    bert_tokens = []
    token_map = []
    tokenizer = FullTokenizer(vocab_file=bert + 'vocab.txt', do_lower_case=False)

    for text in data:
        text_tokens = ['[CLS]']
        text_map = []
        for word in text[0].split(' '):
            text_map.append(len(text_tokens))
            text_tokens.extend(tokenizer.tokenize(word))

        token_map.append(text_map)
        bert_tokens.append(text_tokens)


    if run:
        args = ['python', 'bert/extract_features.py']
        args.append('--input_file=texts')
        args.append('--output_file=WSI')
        args.append('--vocab_file=' + bert + 'vocab.txt')
        args.append('--bert_config_file=' + bert + 'bert_config.json')
        args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
        args.append('--layers=' + layers)
        args.append('--max_seq_length=128')
        args.append('--batch_size=8')
        args.append('--do_lower_case=False')
        args.append('--attention=False')
        args.append('--mask_underscore=False')
        subprocess.run(args)


    targets = [token_map[i][example[1]] for i, example in enumerate(data)]
    labels = [example[2] for example in data]


    layer_cls_metrics = []
    layer_v_metrics = []
    layer_avg_dist_metrics = []
    for layer in range(len(layers.split(','))):
        layer_outputs = np.load('WSI_layer_' + str(layer) + '.npz')
        outputs = np.array([layer_outputs['arr_' + str(x)][target] for x, target in enumerate(targets)]) # arrays of shape (num_texts, num_neurons) 


        try:
            pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
            pca = SupervisedPCARegressor(threshold = 0.2)
            # pca = SupervisedPCARegressor(threshold=0.1)
            pca.fit(outputs, labels)

            pca_outputs = pca.get_transformed_data(outputs)
            print(pca_outputs.shape)
        except:
            pca_outputs = outputs

        class_vectors = [[] for _ in np.unique(labels)]
        for idx, label in enumerate(labels):
            class_vectors[label].append(pca_outputs[idx])

        layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1, validate=True))
        layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
        layer_avg_dist_metrics.append(distance_metrics(class_vectors))


    metrics = []
    for idx, cls_metric in enumerate(layer_cls_metrics):
        metrics.append((f'Layer {idx + 1} : avg_clas_score {cls_metric[0]}, min_val_score {cls_metric[1]}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}, conf_interval {cls_metric[3]}', cls_metric[0], cls_metric[1], layer_v_metrics[idx], layer_avg_dist_metrics[idx], cls_metric[5], cls_metric[7], cls_metric[6], cls_metric[2], cls_metric[3], cls_metric[4], idx))
    unique, counts = np.unique(labels, return_counts=True)
    return metrics, 15, len(data), data[:2], list(zip(unique, counts))

def eval_anaphora(input_file, bert, layers, run=True):
    data = np.load(input_file, allow_pickle=True)# [:1] # array of lists [text, target_word_idx, correct_word_idx]
    with open('100_texts', 'w') as f:
        for example in data:
            f.write(example[0])

    bert_tokens = []
    token_map = []
    tokenizer = FullTokenizer(vocab_file=bert + 'vocab.txt', do_lower_case=False)

    for text in data:
        text_tokens = ['[CLS]']
        text_map = []
        for word in text[0].split(' '):
            text_map.append(len(text_tokens))
            text_tokens.extend(tokenizer.tokenize(word))

        token_map.append(text_map)
        bert_tokens.append(text_tokens)


    if run:
        args = ['python', 'bert/extract_features.py']
        args.append('--input_file=100_texts')
        args.append('--output_file=anaphora')
        args.append('--vocab_file=' + bert + 'vocab.txt')
        args.append('--bert_config_file=' + bert + 'bert_config.json')
        args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
        args.append('--layers=' + layers)
        args.append('--max_seq_length=128')
        args.append('--batch_size=8')
        args.append('--do_lower_case=False')
        args.append('--attention=True')
        args.append('--mask_underscore=False')
        subprocess.run(args)

    # correct = [[int(data[i][2])] for i in range(data.shape[0])]
    correct = [list(map(int, example[2:])) for example in data]
    targets = [int(data[i][1]) for i in range(data.shape[0])]
    layer_metrics = []
    for layer in range(len(layers.split(','))):
        att_data = np.load('anaphora_layer_' + str(layer) + '.npz')
        text_attentions_raw = [att_data['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_heads, num_tokens, num_tokens) 

        # text_attentions = []
        # for idx, text_attention in enumerate(text_attentions_raw):
        #     text_attentions.append(text_attention[:,token_map[idx]][:,:,token_map[idx]])

        text_attentions = text_attentions_raw


        head_metrics = []
        head_count = 12
        if len(layers.split(',')) == 24:
            head_count = 16
        for head in range(head_count):

            head_attentions = [text_attentions[i][head, token_map[i][targets[i]]] for i in range(len(text_attentions))]
            # attention_metric checks if highest attention is to the first token of the correct word, input should include only first tokens of every word
            # attention_metric2 checks if highest attention is to any of the tokens of the correct word, input should include all tokens and list mapping word to its first token
            
            # head_metrics.append(attention_metric(head_attentions, correct))
            head_metrics.append(attention_metric2(head_attentions, correct, token_map))
        layer_metrics.append(head_metrics)

    metrics = []
    for l_idx, layer in enumerate(layer_metrics):
        for h_idx, head in enumerate(layer):
            metrics.append((f'Layer {l_idx} head {h_idx}: {head}', head, l_idx, h_idx))

    metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
    return metrics