from sklearn.svm import LinearSVC
from time import time
from morpho_ru_eval.window_sklearn import nbsvm_pipeline
import numpy as np
import bottleneck as bn
from sklearn.externals import joblib


class NBSVM_Seq:
    def __init__(self, ngram_range, fit_scaler, transform_scaler, C):
        self.make_clf_args = {'ngram_range': ngram_range, 'lowercase': True, 'binary': False, 'min_df': 3,
                              'max_df': 1.0,
                              'caps_features': True, 'extra_features': False, 'fit_scaler': fit_scaler,
                              'transform_scaler': transform_scaler, 'clf': LinearSVC(C=C)}
        self.fnames = []

    def fit(self, X, y):
        self.example = X.sample(1)
        self.base_clf = nbsvm_pipeline(X, **self.make_clf_args)
        print('Base classifier: ', self.base_clf)
        print('Fitting on train set of %d windows...' % len(X))
        st = time()
        pipe = self.base_clf.fit(X, y)
        if hasattr(pipe, 'get_params'):
            params = pipe.get_params()
            if 'mapper' in params:
                tmp = params['mapper'].transform(X.head())
                print('Number of features from mapper: ', tmp.shape[-1])
        print('Fitting done in %d sec.' % (time() - st))
        return self

    def predict(self, X):
        print('Predicting on %d windows...' % len(X))
        st = time()
        y = self.base_clf.predict(X)
        print('Predicting done in %d sec.' % (time() - st))
        return y

    def decision_function(self, X):
        print('Predicting on %d windows...' % len(X))
        st = time()
        y_dec = self.base_clf.decision_function(X)
        print('Predicting done in %d sec.' % (time() - st))
        return y_dec

    # get documents in raw form
    def getDocumentsInRow(self, X):
        x = X.to_string(header=False, index=False, index_names=False).split('\n')
        vals = ['#'.join(ele.split()) for ele in x]
        return list(vals)

    # get feature names
    def get_feature_names(self, vectorizer):
        if self.fnames == []:
            X = self.example
            fnames = []
            for column, vect, _ in vectorizer.features:
                if hasattr(vect, 'get_feature_names'):
                    # use 'column_name[feature_name]' if column vectorizer returns feature names
                    names = ['%s[%s]' % (column, n) for n in vect.get_feature_names()]
                    fnames.extend(names)
                else:  # use 'column_name[feature_number]' if doesn't
                    dim = vect.transform(X[column]).shape[-1]
                    names = ['%s[%d]' % (column, i) for i in range(dim)]
                    fnames.extend(names)
            self.fnames = fnames
        else:
            fnames = self.fnames
        return fnames

    # method getting weigths of ngrams for real and predicted class
    def feature_importances(self, test_X, doc_id, classes, real, pred):
        # get vectorizers from pipeline
        vectorizer1 = self.base_clf.named_steps['mapper']
        vectorizer2 = self.base_clf.named_steps['clf'].estimators_[0].named_steps['mnbscaler']
        doc = test_X.iloc[[doc_id]]

        # make vector of current document
        vectors_test1 = vectorizer1.transform(doc)
        vector_test = vectorizer2.transform(vectors_test1)[0]
        valid = vector_test.nonzero()[1]
        document_vector = vector_test.data
        # document_vector = document_vector.reshape((document_vector.shape[1]))

        fnames = self.get_feature_names(vectorizer1)

        # get list of ngrams from document
        ngrams = list(np.array(fnames)[valid]) + ['BIAS']
        weights = {}
        weights[classes[real]] = []
        weights[classes[pred]] = []

        # get weights of ngrams
        if len(classes) != 2:
            for i in {real, pred}:
                classifier = self.base_clf.named_steps['clf'].estimators_[i].named_steps['clf']
                coefs = classifier.coef_.reshape((classifier.coef_.shape[1]))
                weights[classes[i]] = list(np.multiply(document_vector, coefs[valid])) + [classifier.intercept_]
        else:
            classifier = self.base_clf.named_steps['clf'].estimators_[0].named_steps['clf']
            if real != pred:
                coefs_0 = classifier.coef_.reshape((classifier.coef_.shape[1]))
                weights[classes[1]] = list(np.multiply(document_vector, coefs_0[valid])) + [classifier.intercept_]
                coefs_1 = -classifier.coef_.reshape((classifier.coef_.shape[1]))
                weights[classes[0]] = list(np.multiply(document_vector, coefs_1[valid])) + [-classifier.intercept_]
            else:
                if real == 1:
                    coefs = classifier.coef_.reshape((classifier.coef_.shape[1]))
                    intercept = [classifier.intercept_]
                else:
                    coefs = -classifier.coef_.reshape((classifier.coef_.shape[1]))
                    intercept = [-classifier.intercept_]
                weights[classes[real]] = list(np.multiply(document_vector, coefs[valid])) + intercept

        return weights, ngrams

    # get top positive and negative weights of ngrams for classifier without document
    def feature_analysis_top(self, classes, i, top=25):
        # get vectorizer for names
        vectorizer = self.base_clf.named_steps['mapper']
        fnames = self.get_feature_names(vectorizer)

        weights = {}
        weights[classes[i]] = []

        # get weights of ngrams
        if len(classes) != 2:
            classifier = self.base_clf.named_steps['clf'].estimators_[i].named_steps['clf']
            coefs = classifier.coef_.reshape((classifier.coef_.shape[1]))
            valid = np.concatenate((bn.argpartition(-coefs, top)[:top], bn.argpartition(coefs, top)[:top]), axis=0)
            weights[classes[i]] = coefs[valid]
        else:
            classifier = self.base_clf.named_steps['clf'].estimators_[0].named_steps['clf']
            if i == 1:
                coefs = classifier.coef_.reshape((classifier.coef_.shape[1]))
            else:
                coefs = -classifier.coef_.reshape((classifier.coef_.shape[1]))
            valid = np.concatenate((bn.argpartition(-coefs, top)[:top], bn.argpartition(coefs, top)[:top]), axis=0)
            weights[classes[i]] = coefs[valid]

        ngrams = list(np.array(fnames)[valid])
        return weights, ngrams

    # method for getting classes in classifier's order
    def getClasses(self):
        return self.base_clf.named_steps['clf'].classes_

    def getExamplesForNgram(self, X, y, ngram, maxExampleNum):
        # get vectorizers from pipeline
        vectorizer1 = self.base_clf.named_steps['mapper']
        fnames = self.get_feature_names(vectorizer1)
        idx = fnames.index(ngram)
        vectorizer2 = self.base_clf.named_steps['clf'].estimators_[0].named_steps['mnbscaler']

        # make vector of current document
        vectors_test1 = vectorizer1.transform(X)
        vector_test = vectorizer2.transform(vectors_test1)
        ngramX = X.iloc[vector_test[:, idx].nonzero()[0]]
        values_count = ngramX.y_true.value_counts().to_frame().to_html()
        if len(ngramX) > maxExampleNum:
            ex_num = 'Sampled %d / %d ' % (maxExampleNum, len(ngramX))
            ngramX = ngramX.sample(maxExampleNum)
        else:
            ex_num = 'Totally %d ' % len(ngramX)

        return values_count, ngramX.to_html(), ex_num

    def getDocumentHTML(self, test_X, doc_id):
        dataframeOrNot = True
        return test_X.iloc[[doc_id]].to_html(), dataframeOrNot

    def save(self, clf_path):
        return joblib.dump(self, clf_path)

    def load(self, clf_path):
        return joblib.load(clf_path)
