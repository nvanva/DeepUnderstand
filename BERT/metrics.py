import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

# attentions = list of attention arrays
# correct = list lists of correct word indexes
def attention_metric(attentions, correct):
    correct_predictions = 0
    for i in range(len(correct)):
        if np.argmax(attentions[i]) in correct[i]:
            correct_predictions += 1

    return correct_predictions / len(correct)

def attention_metric2(attentions, correct, token_map_i):
    correct_predictions = 0
    token_map = [[] for _ in range(len(token_map_i))]
    for idx in range(len(correct)):
        token_map[idx] = token_map_i[idx] + [attentions[idx].shape[0] - 1] # if correct[i] is the last element of token_map this line will prevent index error
        # print(i, token_map, correct, np.argmax(attentions[i]))
        for i in range(len(correct[idx])):
            if np.argmax(attentions[idx]) in range(token_map[idx][correct[idx][i]], token_map[idx][correct[idx][i]+1]):
                correct_predictions += 1
                break

    return correct_predictions / len(correct)

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
def centroid_metric(vectors, metric='euclidean'):
    centroids = []
    for vector_class in vectors:
        centroids.append(np.average(vector_class, axis=0))
    centroids = np.array(centroids) # (num_classes, vector_len)
    distances = distance.cdist(centroids, centroids, metric)
    return np.average(distances) # not really average distance but other properties are ok

# input_type = 0 => vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
# input_type = 1 => vectors = array (num_vectors, vector_len)
def cluster_metrics(vectors, y=None, input_type=0):
    if input_type == 0:
        X = vectors[0]
        y = np.zeros(len(vectors[0])) # len(x) == x.shape[0] if x is np.array

        for idx, vec_class in enumerate(vectors[1:]):
            X = np.concatenate((X, vec_class))
            y = np.concatenate((y, np.ones(len(vec_class)) * (idx + 1)))
    else:
        X = vectors

    # clustering = KMeans(n_clusters=len(np.unique(y)))
    # clustering = DBSCAN()   # Все в один кластер
    # clustering = AffinityPropagation()
    # clustering = MeanShift()    # Все в один кластер
    # clustering = SpectralClustering(n_clusters=len(np.unique(y)))   # медленно
    homogeneities = []
    completenesses = []
    vs = []
    linkages = ['ward', 'complete', 'average', 'single']
    affinities = ['euclidean', 'cosine']
    for linkage in linkages:
        for affinity in affinities:
            if linkage == 'ward' and affinity != 'euclidean':
                break

            clustering = AgglomerativeClustering(n_clusters=len(np.unique(y)), affinity=affinity, linkage=linkage)
            clustering.fit(X)
            labels = clustering.labels_
            homogeneities.append(metrics.homogeneity_score(y, labels))
            completenesses.append(metrics.completeness_score(y, labels))
            vs.append(metrics.v_measure_score(y, labels))
    best_idx = np.argmax(vs)
    return homogeneities[best_idx], completenesses[best_idx], vs[best_idx]

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
# input_type = 1 => vectors = array (num_vectors, vector_len), y = array (num_vectors) of labels
def classifier_metric(vectors, y=None, val_vectors=None, val_y=None, input_type=0, validate=False, val_fraction=0.85, fold_count=15):
    if input_type == 0:
        X = vectors[0]
        y = np.zeros(len(vectors[0])) # len(x) == x.shape[0] if x is np.array

        for idx, vec_class in enumerate(vectors[1:]):
            X = np.concatenate((X, vec_class))
            y = np.concatenate((y, np.ones(len(vec_class)) * (idx + 1)))

        if val_vectors == None:
            val_vectors = vectors
            val_X = X
            val_y = y
        else:
            val_X = val_vectors[0]
            val_y = np.zeros(len(val_vectors[0])) # len(x) == x.shape[0] if x is np.array
            for idx, vec_class in enumerate(val_vectors[1:]):
                val_X = np.concatenate((val_X, vec_class))
                val_y = np.concatenate((val_y, np.ones(len(vec_class)) * (idx + 1)))
    else:
        X = vectors
        if val_vectors == None:
            val_X = X
            val_y = y
        else:
            val_X = val_vectors

    if validate:
        sep = int(X.shape[0] * val_fraction)
        X, y = shuffle(X, y, random_state=0)
        val_X = X[sep:]
        val_y = y[sep:]
        X = X[:sep]
        y = y[:sep]

    cross_val_scores = []
    avg_val_score = []
    C_options = [0.01, 0.03, 0.1, 0.3, 1, 3]
    for C in C_options:
        clf = LogisticRegression(max_iter=500, C=C)
        scores = cross_val_score(clf, X, y, cv=fold_count)
        cross_val_scores.append(scores)
        avg_val_score.append(np.average(scores))
        # avg_val_score.append(clf.score(val_X, val_y))

    C = 0.01
    while np.argmax(avg_val_score) == 0:
        C /= 3
        C_options.insert(0, C)
        clf = LogisticRegression(max_iter=500, C=C)
        scores = cross_val_score(clf, X, y, cv=fold_count)
        cross_val_scores.insert(0, scores)
        avg_val_score.insert(0, np.average(scores))

    C = 3
    while np.argmax(avg_val_score) == len(avg_val_score) - 1:
        C *= 3
        C_options.append(C)
        clf = LogisticRegression(max_iter=500, C=C)
        scores = cross_val_score(clf, X, y, cv=fold_count)
        cross_val_scores.append(scores)
        avg_val_score.append(np.average(scores))

    best_avg_score_idx = np.argmax(avg_val_score)
    best_scores = np.sort(cross_val_scores[best_avg_score_idx])
    min_score = best_scores[0]
    max_score = best_scores[-1]
    twenty_percent = np.max([int(fold_count * 0.2), 1])
    conf_interval = [best_scores[twenty_percent], best_scores[-twenty_percent-1]]
    total_interval = [min_score, max_score]

    C_scores = [(C, score) for score, C in zip(avg_val_score, C_options)]

    best_clf = LogisticRegression(max_iter=500, C=C_options[best_avg_score_idx]).fit(X,y)
    train_score = best_clf.score(X,y)

    return avg_val_score[best_avg_score_idx], min_score, C_scores, conf_interval, total_interval, train_score

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
def distance_metrics(vectors, metric='cosine'):
    dif_distances = []
    same_distances = []
    for idx, vecs1 in enumerate(vectors):
        same_distances += list(distance.cdist(vecs1, vecs1, metric).reshape(-1))
        for vecs2 in vectors[idx+1:]:
            dif_distances += list(distance.cdist(vecs1, vecs2, metric).reshape(-1))
    return np.average(same_distances) / np.average(dif_distances)
    # distances = np.array(distances)
    # return np.min(distances), np.max(distances), np.average(distances)