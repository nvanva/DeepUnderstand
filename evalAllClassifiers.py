# coding=utf-8
import pandas as pd
import numpy as np

import sys
from datetime import datetime
import pickle
import json
import os
from sklearn.metrics import f1_score, accuracy_score
from collections import OrderedDict



evalsDir = 'evaluations'
classifiersDir = 'classifiers'
datasetsDir = 'datasets'


def save_clf(clf, pred, train_X, train_y, test_X, test_y, target_names, classifierName):
    savePath = os.path.join(classifiersDir, 'classifiersReady2use', classifierName)
    if not os.path.exists(savePath):
        os.makedirs(savePath, exist_ok=True)

    classes = list(target_names)

    clf_path = os.path.join(savePath, 'clf.pkl')

    if list(train_y)[0] in classes:
            classes = clf.getClasses()

    pred = np.array(pred)
    pred_train = np.array(clf.predict(train_X))
    if hasattr(clf, 'predict_proba'):
        pred_proba = np.array(clf.predict_proba(test_X))
        pred_proba_train = np.array(clf.predict_proba(train_X))
    else:
        pred_proba = np.array(clf.decision_function(test_X))
        pred_proba_train = np.array(clf.decision_function(train_X))


    train_y = np.array(train_y)
    test_y = np.array(test_y)



    clf.save(clf_path)
    with open(os.path.join(savePath, 'pred'), 'wb') as f:
        pickle.dump(pred, f)
    with open(os.path.join(savePath, 'pred_proba'), 'wb') as f:
        pickle.dump(pred_proba, f)
    with open(os.path.join(savePath, 'pred_train'), 'wb') as f:
        pickle.dump(pred_train, f)
    with open(os.path.join(savePath, 'pred_proba_train'), 'wb') as f:
        pickle.dump(pred_proba_train, f)
    with open(os.path.join(savePath, 'test_y'), 'wb') as f:
        pickle.dump(test_y, f)
    with open(os.path.join(savePath, 'test_X'), 'wb') as f:
        pickle.dump(test_X, f)
    with open(os.path.join(savePath, 'classes'), 'wb') as f:
        pickle.dump(classes, f)
    with open(os.path.join(savePath, 'train_y'), 'wb') as f:
        pickle.dump(train_y, f)
    with open(os.path.join(savePath, 'train_X'), 'wb') as f:
        pickle.dump(train_X, f)

    print('%s classifier\'s files saved' % classifierName)


def score_df(y_pred, X, y_true, run_dir=None):
    dd = OrderedDict()
    dd['acc'] = accuracy_score(y_true, y_pred)
    for avg in ['micro', 'macro', 'weighted']:
        dd.update({'f1_'+avg: f1_score(y_true, y_pred, average=avg)})
    return pd.DataFrame(dd, index=[0])

def time_str():
    return ('%s' % datetime.now().replace(microsecond=0)).replace(' ','_')


def evaluate_clf(clf, dataset_proc, eval_function, classifierName, clfBoardname, log_to_file=False, train_parts='train'):
    """
    Evaluate classifier on dataset.
    :param train_limit: use int value to limit train set (usefull for debugging - faster training, but much worse results!)
    :param log_to_file:
    :param clf: object with fit(X, y), predict(X) methods, where X,y - dataset in setences format (list of lists of tokens / tags)
    :param run_label: label of the run, used as prefix for the results file (should not contain characters not allowed in filenames)
    :param only_pos: evaluate only on POS-tags, use only for fast pre-evaluation (final classifier should also return morphological attributes)
    :param only_evaluated: train classifier only on evaluated POSes / attributes; use empty string as tags for non-evaluated
        POSes, filter non-evaluated attributes; results in fewer classes (easier to fit), but less supervised information provided to the classifier
    :return:
    """
    run_dir = './runs/%s-%s/' % (classifierName, time_str())
    os.makedirs(run_dir, exist_ok=True)
    if not log_to_file:
        return evaluate_clf_(clf, dataset_proc, eval_function, run_dir, classifierName, clfBoardname, train_parts=train_parts)

    terminal = sys.stdout
    try:
        with open(run_dir+'.log','w') as sys.stdout:
            return evaluate_clf_(clf, dataset_proc, eval_function, run_dir, classifierName, clfBoardname, train_parts=train_parts)
    finally:
        sys.stdout = terminal


def evaluate_clf_(clf, dataset_proc, eval_function, run_dir, classifierName, clfBoardname, train_parts='train'):
    print(time_str(), 'Loading dataset...')
    train_X, train_y, target_names = dataset_proc(train_parts)
    dev_X, dev_y, _ = dataset_proc('dev')
    print(time_str(), 'fit_predict_score: Fitting on TRAIN set (%d examples) %r...' % (len(train_X),clf))
    clf = clf.fit(train_X, train_y)
    print(time_str(), 'fit_predict_score: Predicting on DEV set (%d examples) %r...' % (len(dev_X),clf))
    pred = clf.predict(dev_X)
    if hasattr(clf, 'feature_importances'):
        save_clf(clf, pred, train_X, train_y, dev_X, dev_y, target_names, classifierName)

    df = eval_function(pred, dev_X, dev_y, run_dir)
    df['clf'] = ('%s' % clfBoardname).replace('\n', ' ')
    df.to_csv(run_dir + 'results.csv', sep='\t', index=False, float_format='%.3lf')

    return df






def transformLeaderboardForVis(data):
    columns = ['Classifier', 'metrics', 'score']
    leaderboardForVis = pd.DataFrame(columns=columns)
    for _, r in data.iterrows():
        for col in data.columns:
            if not col == 'clf':
                row = pd.DataFrame([[r['clf'], col, r[col]]], columns=columns)
                leaderboardForVis = pd.concat([leaderboardForVis, row], axis=0)
    return leaderboardForVis

def loadClf(classifierCard):
    clfModule = classifierCard['module']
    mod = __import__(clfModule, fromlist=[classifierCard['class']])
    classifier = getattr(mod, classifierCard['class'])
    clf = classifier(**classifierCard['params'])
    return clf

def loadClfDataset(classifierCard, datasetsDir):
    classifierName = classifierCard['boardname'] + '_' + classifierCard['dataset']
    clf = loadClf(classifierCard)

    dataset = classifierCard['dataset']
    train_parts = classifierCard['trainset']
    clfBoardname = classifierCard['boardname']

    fpath = os.path.join(datasetsDir, dataset + '.json')
    print('Loading dataset card: ', fpath)
    with open(fpath) as jsonData:
        datasetCard = json.load(jsonData)
    datasetModule = datasetCard['module']
    mod = __import__(datasetModule, fromlist=[datasetCard['load_function']])
    dataset_proc_without_params = getattr(mod, datasetCard['load_function'])
    dataset_proc = lambda x: dataset_proc_without_params(parts=x, **datasetCard['params'])

    evaluation = datasetCard['evaluation_file']
    return classifierName, clf, train_parts, dataset, dataset_proc, evaluation, clfBoardname



def getOneTableForClf(classifierCard, datasetsDir, evalsDir):

    classifierName, clf, train_parts, datasetName, dataset_proc, evaluation, clfBoardname = loadClfDataset(classifierCard, datasetsDir)

    fpath = os.path.join(evalsDir, evaluation)
    print('Loading eval card: ', fpath)
    with open(fpath) as jsonData:
        evalCard = json.load(jsonData)
    evalModule = evalCard['module']
    mod = __import__(evalModule, fromlist=[evalCard['eval_function']])
    eval_function = getattr(mod, evalCard['eval_function'])

    df = evaluate_clf(clf, dataset_proc, eval_function, classifierName, clfBoardname, train_parts=train_parts)
    cur_score = transformLeaderboardForVis(df)
    return cur_score, datasetName, classifierName


def makeLeaderboardDirs(datasetsDir):
    leaderboardPath = {}
    for filename in os.listdir(datasetsDir):
        if filename.endswith(".json"):
            dataset = filename[:filename.find(".json")]
            leaderboardPath[dataset] = os.path.join('leaderboards', dataset)
            if not os.path.exists(leaderboardPath[dataset]):
                os.makedirs(leaderboardPath[dataset], exist_ok=True)
    return leaderboardPath




if __name__ == "__main__":
    leaderboardPath = makeLeaderboardDirs(datasetsDir)
    allJsons = []
    allClassifierNames = []
    for filename in os.listdir(classifiersDir):
        if filename.endswith(".json"):
            allJsons.append(filename)
            print("Loading classifier card: %s" % filename)
            with open(os.path.join(classifiersDir, filename)) as jsonClass:
                classifierCard = json.load(jsonClass)
            if not classifierCard['enabled']:
                print("Classifier card %s is disabled - skipping." % filename)
                continue
            cur_score, datasetName, classifierName = getOneTableForClf(classifierCard, datasetsDir, evalsDir)
            allClassifierNames.append(classifierName)
            cur_score.to_csv(os.path.join(leaderboardPath[datasetName], classifierName + '.csv'), sep='\t', index=False)
    with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allJsons'), 'wb') as f:
        pickle.dump(allJsons, f)
    with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames'), 'wb') as f:
        pickle.dump(allClassifierNames, f)



