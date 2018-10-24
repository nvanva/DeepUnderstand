# coding=utf-8
import os
import argparse
import json
import pickle

from evalAllClassifiers import getOneTableForClf, makeLeaderboardDirs

classifiersDir = 'classifiers'
datasetsDir = 'datasets'
evalsDir = 'evaluations'
leaderboardPath = makeLeaderboardDirs(datasetsDir)
parser = argparse.ArgumentParser()
parser.add_argument("--clf", "--classifier", action="store", help="path to json card of the classifier")
args = parser.parse_args()
filename = args.clf


if __name__ == "__main__":
    if filename.endswith(".json"):
        with open(os.path.join(classifiersDir, filename)) as jsonClass:
            print("Loading classifier card: %s" % filename)
            with open(os.path.join(classifiersDir, filename)) as jsonClass:
                classifierCard = json.load(jsonClass)
            cur_score, datasetName, classifierName = getOneTableForClf(classifierCard, datasetsDir, evalsDir)
            cur_score.to_csv(os.path.join(leaderboardPath[datasetName], classifierName + '.csv'), sep='\t', index=False)


        if os.path.exists(os.path.join(classifiersDir, 'classifiersReady2use', 'allJsons')):
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allJsons'), 'rb') as f:
                allJsons = pickle.load(f)
        else:
            allJsons = []
        if os.path.exists(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames')):
            with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames'), 'rb') as f:
                allClassifierNames = pickle.load(f)
        else:
            allClassifierNames = []

        if classifierName not in allClassifierNames:
            allJsons.append(filename)
            allClassifierNames.append(classifierName)

        with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allJsons'), 'wb') as f:
            pickle.dump(allJsons, f)
        with open(os.path.join(classifiersDir, 'classifiersReady2use', 'allClassifierNames'), 'wb') as f:
            pickle.dump(allClassifierNames, f)
    else:
        print('It\'s not a json file')






