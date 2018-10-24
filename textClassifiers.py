import flask
from flask import Flask, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8


import numpy as np
import pandas as pd
import pickle
import argparse
import json

from bokeh.models import HoverTool, BoxSelectTool, WheelZoomTool, PanTool, ResizeTool, OpenURL, TapTool, SaveTool
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row

import seaborn as sns

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import random
import io
import base64
import os
from werkzeug.contrib.cache import SimpleCache
from evalAllClassifiers import loadClf
cache = SimpleCache(default_timeout=60*60)



app = Flask(__name__)

MAXEXAMPLENUM = 100
MAXDOCUMENTSTOSHOW = 200
TOPFEATURES = 25 #how many most and less weighted features to show
classifiersDir = 'classifiers'
leadDir = 'leaderboards'

class refUrl:
    def __init__(self, url, name):
        self.url = url
        self.name = name

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

def getlist(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj.getlist(item)

def transformLeaderboard(data, datasetName, filename, allClassifierNames):
    classifiers = set(data.Classifier)
    columns = ['Name']

    for clf in classifiers:
        dataForOne = data.loc[data.Classifier == clf]
        for i in dataForOne.iterrows():
            columns.append(i[1]['metrics'])
        columns.append('Analysis')
        break

    clfLink = {}
    for clf in classifiers:
        classifierName = clf + '_' + datasetName
        clfPath = os.path.join(classifiersDir, 'classifiersReady2use', classifierName)
        if os.path.exists(clfPath) and len(list(os.listdir(clfPath))) == 10:
            clfLink[clf] = '<a href="http://{0}:{1}/{2}">Analyze</a>'.format(args.host, args.port, allClassifierNames.index(classifierName))
        else:
            clfLink[clf] = 'Can\'t analyze'

    allClassifiers = pd.DataFrame(columns=columns)
    for clf in classifiers:
        dataForOne = data.loc[data.Classifier == clf]
        metrics = {}
        metrics['Name'] = clf
        for i in dataForOne.iterrows():
            metrics[i[1]['metrics']] = i[1]['score']
        metrics['Analysis'] = clfLink[clf]
        df = pd.DataFrame.from_dict(metrics, orient='index').transpose()
        allClassifiers = pd.concat([allClassifiers, df[columns]], axis=0)

    if not allClassifiers.empty:
        allClassifiers.to_csv(filename, sep='\t', index=False)
        allClassifiers.index = range(len(allClassifiers))
    return allClassifiers


#function for preparing weigths data for plotting
def makeDataFrameForBarplot(weights, ngrams, real_category, pred_category):
    categories = {real_category, pred_category}
    weights_data = []
    for cat in categories:
        if cat == real_category:
            real_pred = ' - real'
        else:
            real_pred = ' - predicted'
        weights_cat = {}
        weights_cat['ngram'] = ngrams
        weights_cat['weight'] = weights[cat]
        weights_cat['category'] = [(cat + real_pred)] * len(ngrams)
        weights_data.append(pd.DataFrame(weights_cat))

    weights_data = pd.concat(weights_data, ignore_index=True)
    return weights_data

#function for redcing dataframe with ngrams to make it more compact
def reduceNgrams(data, order):
    cat1 = list(set(data['category']))[0]
    cat2 = list(set(data['category']))[1]
    weights1 = np.array(data.loc[data['category'] == cat1]['weight'])
    weights2 = np.array(data.loc[data['category'] == cat2]['weight'])
    diff = np.sum(weights1 - weights2)
    weightOther1 = 0
    weightOther2 = 0
    numOfNgrams = len(order)
    mark = False
    deleteFrom = numOfNgrams
    for i, ngram in enumerate(reversed(order)):
        dataForOneNgram = data.loc[data['ngram'] == ngram]
        diffForOneNgram = dataForOneNgram.loc[dataForOneNgram['category'] == cat1].iloc[0]['weight'] - \
                          dataForOneNgram.loc[dataForOneNgram['category'] == cat2].iloc[0]['weight']
        if diff * (diff - diffForOneNgram) < 0 or numOfNgrams - i <= 20:
            break
        else:
            mark = True
            deleteFrom = -(i + 1)
            weightOther1 += dataForOneNgram.loc[dataForOneNgram['category'] == cat1].iloc[0]['weight']
            weightOther2 += dataForOneNgram.loc[dataForOneNgram['category'] == cat2].iloc[0]['weight']
            data = data.loc[data['ngram'] != ngram]
    if mark:
        dataOther = pd.DataFrame([[cat1, 'OTHER_avg[%d]' % abs(deleteFrom), weightOther1 / abs(deleteFrom)], [cat2, 'OTHER_avg[%d]' % abs(deleteFrom), weightOther2 / abs(deleteFrom)]], \
                                 columns=['category', 'ngram', 'weight'])
        print(dataOther)
        data = pd.concat([data, dataOther], axis=0)
        order = order[:deleteFrom] + ['OTHER_avg[%d]' % abs(deleteFrom)]
    return data, order

def get_confusion_matrix(classes, pred, test_y):
    # Confusion matrix
    print('Building data for confusion matrix ...')
    data = pd.DataFrame(columns=['Predicted', 'Real', 'Log of number'])
    maxLog = 0.0
    annot = np.zeros((len(classes), len(classes)), dtype=str).tolist()
    for real_unit in classes:
        t = test_y
        if t[0] in classes:
            i = real_unit
            i_idx = list(classes).index(real_unit)
        else:
            i = list(classes).index(real_unit)
            i_idx = i
        t = pred[t == i]
        for pred_unit in classes:
            if t[0] in classes:
                j = pred_unit
                j_idx = list(classes).index(pred_unit)
            else:
                j = list(classes).index(pred_unit)
                j_idx = j
            err_num = len(t[t == j])
            if err_num == 0:
                cur = pd.DataFrame([[pred_unit, real_unit, 0]], columns=['Predicted', 'Real', 'Log of number'])
            else:
                maxLog = max(maxLog, np.log(err_num))
                cur = pd.DataFrame([[pred_unit, real_unit, np.log(err_num)]],
                                   columns=['Predicted', 'Real', 'Log of number'])
            annot[j_idx][i_idx] = '%.2f%%' % (err_num * 100 / len(test_y))
            data = pd.concat([data, cur], axis=0)
    print('Drawing confusion matrix ...')
    heatmapPath = io.BytesIO()
    data = data.pivot('Predicted', 'Real', 'Log of number')
    hm = sns.heatmap(data, annot=np.array(annot), linewidths=.05, cmap="YlGnBu", fmt='s')
    cbar = hm.collections[0].colorbar
    sns.plt.subplots_adjust(bottom=0.2)
    sns.plt.xticks(rotation=30)
    sns.plt.yticks(rotation=30)
    listOfTicks = np.linspace(0.0, maxLog, 8)
    listOfLabels = ['low'] + ["%d" % int(np.exp(x)) for x in listOfTicks if x != 0.0]
    cbar.set_ticks(listOfTicks)
    cbar.set_ticklabels(listOfLabels)
    fig3 = hm.get_figure()
    fig3.set_size_inches(16, 9)
    fig3.savefig(heatmapPath, bbox_inches='tight', format='png')
    fig3.savefig('hm.eps', bbox_inches='tight', format='eps')
    fig3.clear()
    heatmapPath.seek(0)
    heatmapUrl = base64.b64encode(heatmapPath.getvalue()).decode()
    return heatmapUrl


def get_scores_plot(classes, class_1, class_2, classifier, pred, pred_proba, test_X, test_y, traintest):
    if list(test_y)[0] not in classes:
        id_1 = classes.index(class_1)
        id_2 = classes.index(class_2)
    else:
        id_1 = class_1
        id_2 = class_2
    idc_1 = list(classes).index(class_1)
    idc_2 = list(classes).index(class_2)
    # Plot predictions of 2 classes
    valid_idxs_false = ((pred == id_2) & (test_y == id_1)) | ((pred == id_1) & (test_y == id_2))
    valid_idxs_true = ((pred == id_2) & (test_y == id_2)) | ((pred == id_1) & (test_y == id_1))
    num_doc_false = (pred_proba[valid_idxs_false, idc_2]).shape[0]
    num_doc = (pred_proba[valid_idxs_false | valid_idxs_true, idc_2]).shape[0]

    if num_doc > MAXDOCUMENTSTOSHOW:
        if num_doc_false > MAXDOCUMENTSTOSHOW / 10:
            valid = np.zeros((pred.shape[0]), dtype=bool)
            idxs = random.sample(list(np.array(range(pred.shape[0]))[valid_idxs_false | valid_idxs_true]), MAXDOCUMENTSTOSHOW)
            valid[idxs] = True
            valid_text = 'Sampled %d / %d documents' % (MAXDOCUMENTSTOSHOW, num_doc)
        else:
            valid = np.zeros((pred.shape[0]), dtype=bool)
            idxs = random.sample(list(np.array(range(pred.shape[0]))[valid_idxs_true]), MAXDOCUMENTSTOSHOW - num_doc_false)
            valid[idxs] = True
            valid[valid_idxs_false] = True
            valid_text = 'Sampled %d / %d documents with all false predicted' % (MAXDOCUMENTSTOSHOW, num_doc)
    else:
        valid = np.ones((pred.shape[0]), dtype=bool)
        valid_text = 'Totall %d documents' % num_doc


    print('Prepare data sources ...')
    source10 = ColumnDataSource(
        data=dict(
            x=list((pred_proba[(pred == id_1) & (test_y == id_2) & valid, idc_2]) + (
            np.random.random(len(list(pred_proba[(pred == id_1) & (test_y == id_2) & valid, idc_2]))) - 0.5) / 50),
            y=list((pred_proba[(pred == id_1) & (test_y == id_2) & valid, idc_1]) + (
            np.random.random(len(list(pred_proba[(pred == id_1) & (test_y == id_2) & valid, idc_1]))) - 0.5) / 50),
            desc=[r[:400] + '...' for r in np.array(test_X)[(pred == id_1) & (test_y == id_2) & valid]],
            index=list(np.where((pred == id_1) & (test_y == id_2) & valid)[0])
        )
    )
    source01 = ColumnDataSource(
        data=dict(
            x=list((pred_proba[(pred == id_2) & (test_y == id_1) & valid, idc_2]) + (
            np.random.random(len(list(pred_proba[(pred == id_2) & (test_y == id_1) & valid, idc_2]))) - 0.5) / 50),
            y=list((pred_proba[(pred == id_2) & (test_y == id_1) & valid, idc_1]) + (
            np.random.random(len(list(pred_proba[(pred == id_2) & (test_y == id_1) & valid, idc_1]))) - 0.5) / 50),
            desc=[r[:400] + '...' for r in np.array(test_X)[(pred == id_2) & (test_y == id_1) & valid]],
            index=list(np.where((pred == id_2) & (test_y == id_1) & valid)[0])
        )
    )
    source11 = ColumnDataSource(
        data=dict(
            x=list((pred_proba[(pred == id_1) & (test_y == id_1) & valid, idc_2]) + (
            np.random.random(len(list(pred_proba[(pred == id_1) & (test_y == id_1) & valid, idc_2]))) - 0.5) / 50),
            y=list((pred_proba[(pred == id_1) & (test_y == id_1) & valid, idc_1]) + (
            np.random.random(len(list(pred_proba[(pred == id_1) & (test_y == id_1) & valid, idc_1]))) - 0.5) / 50),
            desc=[r[:400] + '...' for r in np.array(test_X)[(pred == id_1) & (test_y == id_1) & valid]],
            index=list(np.where((pred == id_1) & (test_y == id_1) & valid)[0])
        )
    )
    source00 = ColumnDataSource(
        data=dict(
            x=list((pred_proba[(pred == id_2) & (test_y == id_2) & valid, idc_2]) + (
            np.random.random(len(list(pred_proba[(pred == id_2) & (test_y == id_2) & valid, idc_2]))) - 0.5) / 50),
            y=list((pred_proba[(pred == id_2) & (test_y == id_2) & valid, idc_1]) + (
            np.random.random(len(list(pred_proba[(pred == id_2) & (test_y == id_2) & valid, idc_1]))) - 0.5) / 50),
            desc=[r[:400] + '...' for r in np.array(test_X)[(pred == id_2) & (test_y == id_2) & valid]],
            index=list(np.where((pred == id_2) & (test_y == id_2) & valid)[0])
        )
    )
    hover1 = HoverTool(
        tooltips="""
            <style type="text/css">

                span {
                  display: inline-block;
                  width: 350px;
                }

            </style>
            <div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                </div>
                <div>
                    <span style="font-size: 15px; color: #966;">[@index]</span>
                </div>
                <div>
                    <span style="font-size: 15px;">Location:</span>
                </div>
                <div>
                    <span style="font-size: 10px; color: #696;">(@x, @y)</span>
                </div>
            </div>
            """
    )
    hover2 = HoverTool(
        tooltips="""
            <style type="text/css">

                span {
                  display: inline-block;
                  width: 500px;
                }

            </style>
            <div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                </div>
                <div>
                    <span style="font-size: 15px; color: #966;">[@index]</span>
                </div>
                <div>
                    <span style="font-size: 15px;">Location:</span>
                </div>
                <div>
                    <span style="font-size: 10px; color: #696;">(@x, @y)</span>
                </div>
            </div>
            """
    )
    TOOLS1 = [BoxSelectTool(), WheelZoomTool(), hover1, PanTool(), ResizeTool(), TapTool(), SaveTool()]
    fig1 = figure(plot_width=600, plot_height=600, title="Document predictions for %s classifier" % classifier,
                  x_axis_label=class_2,
                  y_axis_label=class_1, tools=TOOLS1)
    radius = 5
    fig1.circle('x', 'y', legend="True predicted '%s' class" % class_1, size=radius, source=source11, color='blue')
    fig1.circle('x', 'y', legend="True predicted '%s' class" % class_2, size=radius, source=source00, color='green')
    fig1.circle('x', 'y', legend="False predicted '%s' class" % class_1, size=radius, source=source10, color='orange')
    fig1.circle('x', 'y', legend="False predicted '%s' class" % class_2, size=radius, source=source01, color='red')
    classifier1 = traintest + '_' + classifier
    url = 'http://' + args.host + ':%d' % args.port + '/documents/' + classifier1 + '$%$' + '@index'# url for certain document
    taptool = fig1.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    fig1.toolbar.active_scroll = TOOLS1[1]
    TOOLS2 = [BoxSelectTool(), WheelZoomTool(), hover2, PanTool(), ResizeTool(), TapTool(), SaveTool()]
    fig2 = figure(plot_width=600, plot_height=600, title="Document false predictions for %s classifier" % classifier,
                  x_axis_label=class_2,
                  y_axis_label=class_1, tools=TOOLS2)
    radius = 5
    fig2.circle('x', 'y', legend="False predicted '%s' class" % class_1, size=radius, source=source10, color='orange')
    fig2.circle('x', 'y', legend="False predicted '%s' class" % class_2, size=radius, source=source01, color='red')
    url = 'http://' + args.host + ':%d' % args.port + '/documents/' + classifier1 + '$%$' + '@index'# url for certain document
    taptool = fig2.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    fig2.toolbar.active_scroll = TOOLS2[1]
    fig = row([fig1, fig2])
    return fig, valid_text



@app.route("/")
def classifiersCompetition():

    sns.set_style("dark")

    allLeaderboards = []
    plotUrls = []
    datasetNames = []
    with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames'), 'rb') as f:
         allClassifierNames = pickle.load(f)

    for datasetName in os.listdir(leadDir):
        curDir = os.path.join(leadDir, datasetName)
        if os.path.isdir(curDir):
            data = pd.DataFrame(columns=['Classifier', 'metrics', 'score'])
            for filename in os.listdir(curDir):
                if filename.endswith(".csv") and not filename == 'leaderboard.csv':
                    dataForOneClf = pd.read_csv(os.path.join(curDir, filename), sep='\t')
                    data = pd.concat([data, dataForOneClf], axis=0)
            if not data.empty:
                allClassifiers = transformLeaderboard(data, datasetName, os.path.join(curDir, 'leaderboard.csv'), allClassifierNames)
                allLeaderboards.append(allClassifiers.to_html(escape=False))
                datasetNames.append(datasetName)


                barplotMetrics = io.BytesIO()
                numClf = len(set(data.Classifier))
                minY = float(data.min(1).min())
                maxY = float(data.max(1).max())
                if (maxY - minY) < 0.1:
                    minY -= 0.01
                    maxY += 0.01
                else:
                    tmp = (maxY - minY) / 10
                    minY -= tmp
                    maxY += tmp

                bp = sns.barplot(x='metrics', y='score', hue='Classifier', data=data)
                bp.set(ylim=(minY, maxY))
                sns.plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=numClf)
                sns.plt.subplots_adjust(bottom=0.25)
                bp.yaxis.grid()
                fig = bp.get_figure()
                fig.set_size_inches(numClf * 4, 3)
                fig.savefig(barplotMetrics, bbox_inches='tight',  format='png')
                fig.clear()
                barplotMetrics.seek(0)
                plotUrl = base64.b64encode(barplotMetrics.getvalue()).decode()
                plotUrls.append(plotUrl)


    cache.set('allClassifierNames', allClassifierNames)
    html = flask.render_template(
        'leaderboard.html',
        leader_boards=zip(allLeaderboards, plotUrls, datasetNames)
    )
    return encode_utf8(html)


@app.route("/<int:clf_id>")
def plotClassifier(clf_id):

    #Grab classifier names from cache
    allClassifierNames = cache.get('allClassifierNames')
    allJsons = cache.get('allJsons')
    if allClassifierNames is None or allJsons is None:
        with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames'), 'rb') as f:
            allClassifierNames = pickle.load(f)
        with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allJsons'), 'rb') as f:
            allJsons = pickle.load(f)
        cache.set('allClassifierNames', allClassifierNames)
        cache.set('allJsons', allJsons)

    classifier = allClassifierNames[clf_id]
    jsonFile = allJsons[clf_id]
    with open(os.path.join(classifiersDir, jsonFile)) as jsonClass:
        classifierCard = json.load(jsonClass)
    with open(os.path.join(classifiersDir, 'classifiersReady2use', 'curJson'), 'wb') as f:
        pickle.dump(classifierCard, f)
    cache.set('classifierCard', classifierCard)

    #grab classifier and predicted values from cache or from the classifiers's directory
    print('Search for classifier in cache ...')
    loadPath = os.path.join(classifiersDir, 'classifiersReady2use', classifier)
    Clf = cache.get(classifier + '_Clf')
    pred = cache.get(classifier + '_pred')
    pred_proba = cache.get(classifier + '_pred_proba')
    pred_train = cache.get(classifier + '_pred_train')
    pred_proba_train = cache.get(classifier + '_pred_proba_train')
    test_y = cache.get(classifier + '_test_y')
    test_X = cache.get(classifier + '_test_X')
    train_y = cache.get(classifier + '_train_y')
    train_X = cache.get(classifier + '_train_X')
    classes = cache.get(classifier + '_classes')
    old_classifier = cache.get(classifier + '_old_classifier')

    if Clf is None or pred is None or pred_proba is None or test_y is None or\
                    test_X is None or classes is None or train_X is None or\
                    train_y is None or pred_train is None or pred_proba_train is None or\
                    old_classifier != classifier:
        print('Load classifier from files ...')


        Clf = loadClf(classifierCard)
        Clf = Clf.load(os.path.join(loadPath, 'clf.pkl'))
        with open(os.path.join(loadPath, 'pred'), 'rb') as f:
            pred = pickle.load(f)
        with open(os.path.join(loadPath, 'pred_proba'), 'rb') as f:
            pred_proba = pickle.load(f)
        with open(os.path.join(loadPath, 'test_y'), 'rb') as f:
            test_y = pickle.load(f)
        with open(os.path.join(loadPath, 'test_X'), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(loadPath, 'pred_train'), 'rb') as f:
            pred_train = pickle.load(f)
        with open(os.path.join(loadPath, 'pred_proba_train'), 'rb') as f:
            pred_proba_train = pickle.load(f)
        with open(os.path.join(loadPath, 'train_y'), 'rb') as f:
            train_y = pickle.load(f)
        with open(os.path.join(loadPath, 'train_X'), 'rb') as f:
            train_X = pickle.load(f)
        with open(os.path.join(loadPath, 'classes'), 'rb') as f:
            classes = pickle.load(f)

        cache.set(classifier + '_Clf', Clf)
        cache.set(classifier + '_test_X', test_X)
        cache.set(classifier + '_test_y', test_y)
        cache.set(classifier + '_pred', pred)
        cache.set(classifier + '_pred_proba', pred_proba)
        cache.set(classifier + '_train_X', train_X)
        cache.set(classifier + '_train_y', train_y)
        cache.set(classifier + '_pred_train', pred_train)
        cache.set(classifier + '_pred_proba_train', pred_proba_train)
        cache.set(classifier + '_old_classifier', classifier)
        cache.set(classifier + '_classes', classes)

        print('Classifier loaded.')

    allcats = []
    for cat in classes:
        allcats.append(refUrl('http://' + args.host + ':%d' % args.port + '/featureAnalysis/' + classifier + '$%$' + cat, cat))
    clfUrl = refUrl('http://' + args.host + ':%d' % args.port + '/errorAnalysis/' + classifier, 'Error analysis')

    #Train/test performance
    acc = accuracy_score(test_y, pred)
    if pred[0] in classes:
        labels = classes
    else:
        labels = range(len(classes))
    test_scores_df = pd.DataFrame({'accuracy': acc, 'error rate': 1.-acc, 'support': len(test_y), 'error number': (1.-acc) * len(test_y)}, index=[0])
    acc_train = accuracy_score(train_y, pred_train)
    train_scores_df = pd.DataFrame({'accuracy': acc_train, 'error rate': 1.-acc_train, 'support': len(train_y), 'error number': (1.-acc_train) * len(train_y)}, index=[0])
    prfs = precision_recall_fscore_support(test_y, pred, labels=labels, average=None)
    classification_report_test = pd.DataFrame({'class': np.array(classes), 'precision': prfs[0], 'recall': prfs[1], 'f1_score': prfs[2], 'support': prfs[3]})
    prfs = precision_recall_fscore_support(train_y, pred_train, labels=labels, average=None)
    classification_report_train = pd.DataFrame({'class': np.array(classes), 'precision': prfs[0], 'recall': prfs[1], 'f1_score': prfs[2], 'support': prfs[3]})




    heatmapUrl = get_confusion_matrix(classes, pred, test_y)

    html = flask.render_template(
        'plots.html',
        hm_url=heatmapUrl,
        allcats=allcats,
        classifier=classifier,
        test_scores=test_scores_df.to_html(),
        train_scores=train_scores_df.to_html(),
        classification_report_test=classification_report_test.to_html(),
        classification_report_train=classification_report_train.to_html(),
        clf=clfUrl
    )
    return encode_utf8(html)


@app.route('/featureAnalysis/<string:clf_cat>')
def featureAnalysis(clf_cat):
    classifier = clf_cat.split('$%$')[0]
    cat = clf_cat.split('$%$')[1]

    classes = cache.get(classifier + '_classes')
    Clf = cache.get(classifier + '_Clf')

    if len(classes) == 1:
        return "Constant predictor!"

    if Clf is None or classes is None:
        loadPath = os.path.join(classifiersDir, 'classifiersReady2use', classifier)
        classifierCard = cache.get('classifierCard')

        if classifierCard is None:
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'curJson'), 'rb') as f:
                classifierCard = pickle.load(f)
            cache.set('classifierCard', classifierCard)
        Clf = loadClf(classifierCard)
        Clf = Clf.load(os.path.join(loadPath, 'clf.pkl'))
        with open(os.path.join(loadPath, 'classes'), 'rb') as f:
            classes = pickle.load(f)
        cache.set(classifier + '_classes', classes)
        cache.set(classifier + '_Clf', Clf)

    if not hasattr(Clf, 'feature_analysis_top'):
        return 'No feature analysis!'
    id_class = list(classes).index(cat)
    weights, ngrams = Clf.feature_analysis_top(list(classes), id_class, top=TOPFEATURES)
    data = makeDataFrameForBarplot(weights, ngrams, cat, cat)
    cnt_ngrams = len(data)

    data = data.sort_values(by='weight', axis=0, ascending=False)
    order = list(data['ngram'])
    barplotWeights = io.BytesIO()
    bp = sns.barplot(y='ngram', x='weight', order=order, hue='category', orient='h', data=data)
    bp.legend_.remove()
    bp.yaxis.grid()
    bp.xaxis.grid()
    fig = bp.get_figure()
    fig.set_size_inches(11, len(data) * 0.5)
    fig.savefig(barplotWeights, bbox_inches='tight', format='png')
    fig.clear()
    barplotWeights.seek(0)
    plotUrl = base64.b64encode(barplotWeights.getvalue()).decode()

    ngramsWithUrls = []
    if hasattr(Clf, 'getExamplesForNgram'):
        for ngram in order:
            ngramsWithUrls.append(refUrl('http://' + args.host + ':%d' % args.port + '/ngrams/' + ngram + '$%$' + classifier, ngram))

    html = flask.render_template(
        'featureAnalysis.html',
        bar_url=plotUrl,
        cat=cat,
        ngrams=ngramsWithUrls,
        top_features=TOPFEATURES
    )
    return encode_utf8(html)


@app.route('/errorAnalysis/<string:classifier>')
def errorAnalysis(classifier):

    # Grab the inputs arguments from the URL
    args_in = request.args

    traintest = getitem(args_in, 'set', 'dev')

    #grab classifier and predicted values from cache or from the classifiers's directory
    print('Search for predicted values in cache ...')
    loadPath = os.path.join(classifiersDir, 'classifiersReady2use', classifier)
    if traintest == 'dev':
        pred = cache.get(classifier + '_pred')
        pred_proba = cache.get(classifier + '_pred_proba')
        test_y = cache.get(classifier + '_test_y')
        test_X = cache.get(classifier + '_test_X')
    else:
        pred = cache.get(classifier + '_pred_train')
        pred_proba = cache.get(classifier + '_pred_proba_train')
        test_y = cache.get(classifier + '_train_y')
        test_X = cache.get(classifier + '_train_X')
    Clf = cache.get(classifier + '_Clf')
    classes = cache.get(classifier + '_classes')
    old_classifier = cache.get(classifier + '_old_classifier')

    if Clf is None or pred is None or pred_proba is None or test_y is None or\
    test_X is None or classes is None or old_classifier != classifier:
        print('Load predicted values ...')
        classifierCard = cache.get('classifierCard')

        if classifierCard is None:
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'curJson'), 'rb') as f:
                classifierCard = pickle.load(f)
            cache.set('classifierCard', classifierCard)
        Clf = loadClf(classifierCard)
        Clf = Clf.load(os.path.join(loadPath, 'clf.pkl'))
        with open(os.path.join(loadPath, 'classes'), 'rb') as f:
            classes = pickle.load(f)

        if traintest == 'dev':
            with open(os.path.join(loadPath, 'pred'), 'rb') as f:
                pred = pickle.load(f)
            with open(os.path.join(loadPath, 'pred_proba'), 'rb') as f:
                pred_proba = pickle.load(f)
            with open(os.path.join(loadPath, 'test_y'), 'rb') as f:
                test_y = pickle.load(f)
            with open(os.path.join(loadPath, 'test_X'), 'rb') as f:
                test_X = pickle.load(f)
            cache.set(classifier + '_test_X', test_X)
            cache.set(classifier + '_test_y', test_y)
            cache.set(classifier + '_pred', pred)
            cache.set(classifier + '_pred_proba', pred_proba)
        else:
            with open(os.path.join(loadPath, 'pred_train'), 'rb') as f:
                pred = pickle.load(f)
            with open(os.path.join(loadPath, 'pred_proba_train'), 'rb') as f:
                pred_proba = pickle.load(f)
            with open(os.path.join(loadPath, 'train_y'), 'rb') as f:
                test_y = pickle.load(f)
            with open(os.path.join(loadPath, 'train_X'), 'rb') as f:
                test_X = pickle.load(f)
            cache.set(classifier + '_train_X', test_X)
            cache.set(classifier + '_train_y', test_y)
            cache.set(classifier + '_pred_train', pred)
            cache.set(classifier + '_pred_proba_train', pred_proba)

        cache.set(classifier + '_Clf', Clf)
        cache.set(classifier + '_old_classifier', classifier)
        cache.set(classifier + '_classes', classes)

        print('Classifier loaded.')

    # get classes for visualize


    if hasattr(Clf, 'getDocumentsInRow'):
        test_X = np.array(Clf.getDocumentsInRow(test_X))
    else:
        test_X = np.array(test_X)
    class_1 = getitem(args_in, '1st_class', classes[0])
    if len(classes) > 1:
        class_2 = getitem(args_in, '2nd_class', classes[1])
    else:
        class_2 = getitem(args_in, '2nd_class', classes[0])

    if class_1 not in classes:
        class_1 = classes[0]
        if len(classes) > 1:
            class_2 = classes[1]
        else:
            class_2 = classes[0]

    if len(classes) == 2 and hasattr(Clf, 'decision_function'):
        pred_proba = np.vstack((-pred_proba.T, pred_proba.T)).T
    elif len(classes) == 2 and hasattr(Clf, 'pred_proba'):
        pred_proba = np.vstack((1 - pred_proba.T, pred_proba.T)).T
    try:
        fig, valid_text = get_scores_plot(classes, class_1, class_2, classifier, pred, pred_proba, test_X, test_y, traintest)
        script, div = components(fig)
    except:
        class_1, class_2, script, div, valid_text = '', '', '', '', 'Nothing'

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()


    html = flask.render_template(
        'errorAnalysis.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
        allcats=classes,
        chosen_1class=class_1,
        chosen_2class=class_2,
        traintest=traintest,
        valid_text=valid_text
    )
    return encode_utf8(html)


@app.route('/documents/<string:clf_id>')
def presentText(clf_id):
    traintest = clf_id.split('$%$')[0].split('_')[0]
    classifier = clf_id.split('$%$')[0][len(traintest) + 1:]
    document_id = int(clf_id.split('$%$')[1])
    top = 25

    classes = cache.get(classifier + '_classes')
    Clf = cache.get(classifier + '_Clf')

    if len(classes) == 1:
        return "Constant predictor!"

    if traintest == 'dev':
        test_X = cache.get(classifier + '_test_X')
        test_y = cache.get(classifier + '_test_y')
        preds = cache.get(classifier + '_pred')
    else:
        test_X = cache.get(classifier + '_train_X')
        test_y = cache.get(classifier + '_train_y')
        preds = cache.get(classifier + '_pred_train')

    if Clf is None or classes is None or test_X is None or test_y is None or preds is None:
        loadPath = os.path.join(classifiersDir, 'classifiersReady2use', classifier)
        classifierCard = cache.get('classifierCard')

        if classifierCard is None:
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'curJson'), 'rb') as f:
                classifierCard = pickle.load(f)
            cache.set('classifierCard', classifierCard)
        Clf = loadClf(classifierCard)
        Clf = Clf.load(os.path.join(loadPath, 'clf.pkl'))
        with open(os.path.join(loadPath, 'classes'), 'rb') as f:
            classes = pickle.load(f)

        if traintest == 'dev':
            with open(os.path.join(loadPath, 'pred'), 'rb') as f:
                preds = pickle.load(f)
            with open(os.path.join(loadPath, 'test_y'), 'rb') as f:
                test_y = pickle.load(f)
            with open(os.path.join(loadPath, 'test_X'), 'rb') as f:
                test_X = pickle.load(f)
            cache.set(classifier + '_test_X', test_X)
            cache.set(classifier + '_test_y', test_y)
            cache.set(classifier + '_pred', preds)
        else:
            with open(os.path.join(loadPath, 'pred_train'), 'rb') as f:
                preds = pickle.load(f)
            with open(os.path.join(loadPath, 'train_y'), 'rb') as f:
                test_y = pickle.load(f)
            with open(os.path.join(loadPath, 'train_X'), 'rb') as f:
                test_X = pickle.load(f)

            cache.set(classifier + '_train_X', test_X)
            cache.set(classifier + '_train_y', test_y)
            cache.set(classifier + '_pred_train', preds)

        cache.set(classifier + '_Clf', Clf)
        cache.set(classifier + '_classes', classes)
    real = test_y[document_id]
    pred = preds[document_id]
    if real in classes:
        real = list(classes).index(real)
        pred = list(classes).index(pred)
    weights, ngrams = Clf.feature_importances(test_X, document_id, list(classes), real, pred)
    data = makeDataFrameForBarplot(weights, ngrams, classes[real], classes[pred])

    barplotWeights = io.BytesIO()
    if real != pred:
        cat1 = list(set(data['category']))[0]
        cat2 = list(set(data['category']))[1]
        weights1 = np.array(data.loc[data['category'] == cat1]['weight'])
        weights2 = np.array(data.loc[data['category'] == cat2]['weight'])
        diff = np.abs(weights1 - weights2)
        ordered_data = pd.DataFrame({'ngram': ngrams, 'difference': diff})
        ordered_data = ordered_data.sort_values(by='difference', axis=0, ascending=False)
        order = list(ordered_data['ngram'])
        data, order = reduceNgrams(data, order)
        oneclass = False
        cnt_ngrams = len(order)
        topText = "Hey, this string won't be shown!"
    else:
        data = data.sort_values(by='weight', axis=0, ascending=False)
        if 2 * top < len(data):
            order = list(data['ngram'])[:top] + list(data['ngram'])[-top:]
            topText = "%d most and least weighted ngrams" % top
        else:
            order = list(data['ngram'])
            topText = "All ngrams"
        oneclass = True
        cnt_ngrams = len(order)

    bp = sns.barplot(y='ngram', x='weight', order=order, hue='category', orient='h', data=data)
    sns.plt.legend(bbox_to_anchor=(1.35, 1))
    sns.plt.subplots_adjust(right=0.75)
    bp.yaxis.grid()
    bp.xaxis.grid()
    fig = bp.get_figure()
    fig.set_size_inches(11, cnt_ngrams * 0.5)
    fig.savefig(barplotWeights, bbox_inches='tight', format='png')
    fig.clear()
    barplotWeights.seek(0)
    plotUrl = base64.b64encode(barplotWeights.getvalue()).decode()
    document_text, dataframeOrNot = Clf.getDocumentHTML(test_X, document_id)
    ngramsWithUrls = []
    if hasattr(Clf, 'getExamplesForNgram'):
        for ngram in order:
            if not ngram.startswith("OTHER"):
                ngramsWithUrls.append(refUrl('http://' + args.host + ':%d' % args.port + '/ngrams/' + ngram + '$%$' + classifier, ngram))
    html = flask.render_template(
        'document.html',
        document_text=document_text,
        dataframe=dataframeOrNot,
        bar_url=plotUrl,
        ngrams=ngramsWithUrls,
        top=topText,
        oneclass=oneclass
    )
    return encode_utf8(html)


@app.route('/ngrams/<string:ngram_clf>')
def presentNgram(ngram_clf):
    ngramName = ngram_clf.split('$%$')[0]
    classifier = ngram_clf.split('$%$')[1]
    loadPath = os.path.join(classifiersDir, 'classifiersReady2use', classifier)
    train_X = cache.get(classifier + '_train_X')
    train_y = cache.get(classifier + '_train_y')
    Clf = cache.get(classifier + '_Clf')
    if train_X is None or train_y is None or Clf is None:
        classifierCard = cache.get('classifierCard')

        if classifierCard is None:
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'curJson'), 'rb') as f:
                classifierCard = pickle.load(f)
            cache.set('classifierCard', classifierCard)
        Clf = loadClf(classifierCard)
        Clf = Clf.load(os.path.join(loadPath, 'clf.pkl'))
        cache.set(classifier + '_Clf', Clf)
        with open(os.path.join(loadPath, 'train_y'), 'rb') as f:
            train_y = pickle.load(f)
        with open(os.path.join(loadPath, 'train_X'), 'rb') as f:
            train_X = pickle.load(f)
        cache.set(classifier + '_train_X', train_X)
        cache.set(classifier + '_train_y', train_y)

    value_counts, windows, ex_num = Clf.getExamplesForNgram(train_X, train_y, ngramName, MAXEXAMPLENUM)

    html = flask.render_template(
        'ngram.html',
        examples=windows,
        value_counts=value_counts,
        ngram_name=ngramName,
        ex_num=ex_num
    )
    return encode_utf8(html)



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', type=int, default=5000)
parser.add_argument('-d', '--debug', action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.debug)
