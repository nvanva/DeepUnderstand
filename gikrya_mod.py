from morpho_ru_eval.utils.gikrya import pos_sentences, posattrs_sentences, POS_ATTRNAME
from morpho_ru_eval.utils.converters import sentences2df

def pos_sentences_mod(parts='dev', only_evaluated=False, attr=POS_ATTRNAME, window=[2,2], limit=None):
    if parts == 'flat':
        words, tags = posattrs_sentences(parts='dev', only_evaluated=only_evaluated, limit=limit, limit_attrs=[attr])
        return words, tags, []
    words, tags = posattrs_sentences(parts=parts, only_evaluated=only_evaluated, limit=limit, limit_attrs=[attr])
    X, _ = to_df(words, tags, window)
    classes = list(X.y_true.unique())
    return X, X.y_true, classes

def to_df(sentences, paths, window):
    print('Converting to DataFrame of windows...')
    n_tokens = 0
    for sent in sentences:
        n_tokens += len(sent)
    X = sentences2df(sentences, paths, win=window,
                           filter_empty=True, nopadding=False, Extra=None)  # skip empty labels (don't train on them)
    print('%d sentences with %d tokens converted to %d windows.' % (len(sentences), n_tokens, len(X)))
    return X, n_tokens