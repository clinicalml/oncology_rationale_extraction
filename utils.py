from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Concatenates a set of notes, such that copy-forwarded text does not appear
# Notes is a list of strings to be concatenated
# Feature_index is the first index of the notes that should be included in our feature_window
# Usage: new_note = get_new_text(notes, feature_index)
def get_new_text(notes, feature_index):
    seen_lines = set()
    new_note = ""
    for i, note in enumerate(notes):
        if i >= feature_index:
            new_note += "\n".join([line for line in note if line not in seen_lines])
            seen_lines.update(set(note))
        else:
            seen_lines.update(set(note))
    if len(new_note) == 0:
        new_note = "blank note"
    return new_note


## Vectorizes notes for input to logistic regression
## Train, val, and test notes are assumed to each be lists of strings
def vectorize_notes(vectorizer_value, train_notes, val_notes, test_notes):
    if vectorizer_value == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10, binary=True)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, binary=True)
    x_train = vectorizer.fit_transform(train_notes).toarray()
    x_val = vectorizer.transform(val_notes).toarray()
    x_test = vectorizer.transform(test_notes).toarray()
    return x_train, x_val, x_test, vectorizer

# Calculates the standard deviation for an AUC
# Assumes true and pred are the ground truth and prediction vectors of equal size
def auc_std(true, pred):
    auc = roc_auc_score(true, pred)
    n1 = sum(true)
    n2 = len(true) - sum(true)
    auc_var = (
        auc*(1-auc) +
        (n1 - 1) * ((
            auc / (2 - auc)
        ) - auc * auc) +
        (n2 - 1) * ((
            2 * auc * auc / (1 + auc)
        ) - auc * auc)
    ) / (n1 * n2)
    return auc_var ** 0.5