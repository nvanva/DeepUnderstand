from sklearn.datasets import fetch_20newsgroups

def fetch_20newsgroups_mod(parts, remove, categories):
    if parts == 'dev':
        test_set = fetch_20newsgroups(subset='test', remove=remove, categories=categories)
        return test_set.data, test_set.target, test_set.target_names
    else:
        train_set = fetch_20newsgroups(subset=parts, remove=remove, categories=categories)
        return train_set.data, train_set.target, train_set.target_names