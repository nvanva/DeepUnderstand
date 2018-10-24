from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.externals import joblib

class VeClf(Pipeline):
    def __init__(self, alpha, ngram_range=(1,1), min_df=1):
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        classifier = MultinomialNB()
        super(VeClf, self).__init__([('Vec', vectorizer), ('Clf', classifier)])
        self.set_params(Clf__alpha=alpha)

    def feature_importances(self, test_X, document_id, classes, real, pred):
        document = test_X[document_id]
        real_category = classes[real]
        pred_category = classes[pred]
        categories = [real_category, pred_category]
        vectorizer = self.named_steps['Vec']
        vectors_test = vectorizer.transform([document]).toarray()
        document_vector = vectors_test[0].T
        classifier = self.named_steps['Clf']
        feature_names = np.asarray(vectorizer.get_feature_names())
        name2id = {}
        for i, name in enumerate(feature_names):
            name2id[name] = i
        analyze = vectorizer.build_analyzer()
        words = set(analyze(document))
        features = []
        for word in words:
            if word in feature_names:
                features.append(word)

        weights={}
        weights[real_category] = []
        weights[pred_category] = []


        for r in features:
            for cat in set(categories):
                weights[cat].append(document_vector[name2id[r]] * classifier.feature_log_prob_[classes.index(cat)][name2id[r]])
        for cat in set(categories):
            weights[cat].append(classifier.class_log_prior_[classes.index(cat)])
        features.append('BIAS')
        return weights, features

    def getDocumentHTML(self, test_X, document_id):
        dataframeOrNot = False
        return test_X[document_id], dataframeOrNot
    def save(self, clf_path):
        return joblib.dump(self, clf_path)
    def load(self, clf_path):
        return joblib.load(clf_path)
