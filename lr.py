# Import scikit_learn dependencies
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight


# Defines and fits logistic regression model for binary tasks (i.e., distinguishing between 2 classes)
# Takes arguments for vectorizer_value (Count vs TFIDF) and data sets with paired note features (_notes) and labels (y_)
def run_binary_LR(vectorizer_value, solver_value, train_notes, y_train, val_notes, y_val, test_notes, y_test):

    # Convert raw note features to vectors
    x_train, x_val, x_test, vectorizer = vectorize_notes(vectorizer_value, train_notes, val_notes, test_notes)

    # Define model and grid search hyperparameters
    lr_param_grid = {'C': [0.01, 0.1, 1], 'penalty': ['l1', 'l2']}
    model_to_fit = GridSearchCV(LogisticRegression(random_state=0, solver='liblinear'), \
                                param_grid=lr_param_grid, scoring='roc_auc', cv=5)

    # Return model for learning curve assessment
    learning_curve_model = model_to_fit

    # Conduct grid search to fit best model
    model_to_fit.fit(x_train, y_train)
    model = model_to_fit.best_estimator_
    params = model_to_fit.best_params_

    # Get label predictions
    y_val_preds = model.predict(x_val)
    y_val_probs = model.predict_proba(x_val)[:, 1]

    # Get cross validation and validation set statistics
    cross_val = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
    cross_val_auc = cross_val.mean()
    cross_val_sd = cross_val.std()
    val_auc = roc_auc_score(y_val, y_val_probs)
    val_sd = auc_std(y_val, y_val_probs)

    return y_val, y_val_preds, y_val_probs, val_auc, val_sd, cross_val_auc, cross_val_sd, params, model,\
                learning_curve_model, vectorizer, x_train

# Defines and fits logistic regression model for multiclass tasks
# Takes arguments for vectorizer_value (Count vs TFIDF) and data sets with paired note features (_notes) and labels (y_)
def run_multiclass_LR(vectorizer_value, solver_value, train_notes, y_train, val_notes, y_val, test_notes, y_test):

    # Convert raw note features to vectors
    x_train, x_val, x_test, vectorizer = vectorize_notes(vectorizer_value, train_notes, val_notes, test_notes)

    # Define model
    model = LogisticRegression(random_state=0, solver=solver_value, multi_class='multinomial', \
                       C=0.01, penalty='l2', class_weight='balanced')

    # Return model for learning curve assessment
    learning_curve_model = model

    # Fit model
    model.fit(x_train, y_train)

    # Get label predictions
    y_val_preds = model.predict(x_val)
    y_val_probs = model.predict_proba(x_val)

    # Get cross validation and validation set statistics
    cross_val = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc_ovr')
    cross_val_auc = cross_val.mean()
    cross_val_sd = cross_val.std()
    cross_val_w = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc_ovr_weighted')
    cross_val_auc_w = cross_val_w.mean()
    cross_val_sd_w = cross_val_w.std()
    val_auc = roc_auc_score(y_val, y_val_probs, multi_class='ovr')
    val_auc_w = roc_auc_score(y_val, y_val_probs, multi_class='ovr', average='weighted')
    val_sd = "Not calculated"
    val_sd_w = "Not calculated"

    return y_val, y_val_preds, y_val_probs, val_auc, val_sd, cross_val_auc, cross_val_sd, val_auc_w, val_sd_w,\
                cross_val_auc_w, cross_val_sd_w, model, learning_curve_model, vectorizer, x_train

