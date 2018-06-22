import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib

train_features = ['code/files','comment/code','test/code','readme/code','docstring/code',
                  'E1/code','E2/code','E3/code','E4/code','E5/code','E7/code',
                  'W1/code','W2/code','W3/code','W6/code','code_lines']

def make_features(df):
    df['code/files'] = df['code_lines']/df['n_pyfiles']
    df['comment/code'] = df['comment_lines']/df['code_lines']
    df['test/code'] = df['test_lines']/df['code_lines']
    df['readme/code'] = df['readme_lines']/df['code_lines']
    df['docstring/code'] = df['docstring_lines']/df['code_lines']
    
    try:
        df['commits/code'] = df['n_commits']/df['code_lines']
    except:
        print('couldnt find n_commits')
    
    for p in ['E1','E2','E3','E4','E5','E7','E9','W1','W2','W3','W5','W6']:
        df['%s/code'%p] = df[p]/df['code_lines']
    return df.dropna(how='any').drop_duplicates()

def load_data(good_dir, bad_dir):
    good_names = ['url', 'n_pyfiles', 'code_lines', 'comment_lines', 'docstring_lines',
                  'test_lines','readme_lines', 'n_commits', 'commits_per_time', 'n_stars',
                  'n_forks', 'E1','E2','E3','E4','E5','E7','E9','W1','W2','W3','W5','W6']
    df_good = pd.read_csv(good_dir, names=good_names)
    df_good = make_features(df_good)

    bad_names = ['url', 'n_pyfiles', 'code_lines', 'comment_lines', 'docstring_lines',
                 'test_lines','readme_lines','E1','E2','E3',
                 'E4','E5','E7','E9','W1','W2','W3','W5','W6']
    df_bad = pd.read_csv(bad_dir, names=bad_names)
    df_bad = make_features(df_bad)
    return df_good, df_bad

def prepare_data(good_dir, bad_dir):
    df_good, df_bad = load_data(good_dir, bad_dir)

    # log data, really useful
    X = np.log10(df_good[train_features])
    Xb = np.log10(df_bad[train_features])

    # replace log10(0) values with min val - 1 order of mag lower
    minvals = {}
    X_join = pd.concat([X, Xb], axis=0)
    for c in X_join.columns:
        minval = np.floor(np.min(X_join.loc[X_join[c] > -np.inf, c].values) - 1)
        X.loc[X[c] == -np.inf, c] = minval
        X.loc[X[c] == np.inf, c] = minval
        Xb.loc[Xb[c] == -np.inf, c] = minval
        Xb.loc[Xb[c] == np.inf, c] = minval
        minvals[c] = minval

    # standardize
    scaler = StandardScaler()
    scaler.fit(pd.concat([X, Xb], axis=0)) # need to scale over all X+Xb data together!
    X = scaler.transform(X)
    Xb = scaler.transform(Xb)
    scaler_name = 'models/scaler.pkl'
    minvals_name = 'models/minvals.pkl'
#    with open(scaler_name, 'wb') as output_:
#        pickle.dump(scaler, output_)
#    with open(minvals_name, 'wb') as output_:
#        pickle.dump(minvals, output_)
    joblib.dump(scaler, scaler_name)
    joblib.dump(minvals, minvals_name)
    return X, Xb

# metric: try to maximize recall whilst including as few background samples as possible
# ref: W. S. Lee and B. Liu, 'Learning with positive and unlabeled examples
# using weighted logistic regression'
def focal_score(y_pred_test, y_pred_bkgnd, nu, gamma):
    # recall
    recall = len(np.where(y_pred_test == 1)[0])/float(len(y_pred_test))
    
    # fraction of background samples with a positive classification
    bckgnd_focal_frac = len(np.where(y_pred_bkgnd == 1)[0])/float(len(y_pred_bkgnd))
    
    try:
        score = recall**2 / bckgnd_focal_frac
    except ZeroDivisionError:
        print(("recall=%f, background_focal_frac=%f,"
               "nu=%f, gamma=%f")%(recall, bckgnd_focal_frac, nu, gamma))
        score = 0
    return score, recall, bckgnd_focal_frac

def random_train_test_split(X, train_frac = 0.8):
    N = len(X)
    rN = np.arange(0, N)
    np.random.shuffle(rN)  # randomly shuffle data
    train_i, test_i = rN[0: int(train_frac*N)], rN[int(train_frac*N):]
    
    X_train, X_test = X[train_i], X[test_i]
    return X_train, X_test

def train_model(X, Xb, nu, loggamma, n_cv=3, recall_thresh=0.80):
    # iterate over hypers, cv
    scores = []
    nu_best = 0
    loggamma_best = 0
    score_best = 0
    for n in nu:
        for g in loggamma:
            sc, rc, bg = [], [], []
            for i in range(n_cv):
                X_train, X_test = random_train_test_split(X)
                Xb_train, Xb_test = random_train_test_split(Xb)
                
                clf = svm.OneClassSVM(kernel='rbf', nu=n, gamma=10**g)
                clf.fit(X_train)
                y_pred_test = clf.predict(X_test)
                y_pred_bkgnd = clf.predict(Xb_train)
                score_, recall_, bkgnd_ = focal_score(y_pred_test, y_pred_bkgnd, n, g)
                sc.append(score_)
                rc.append(recall_)
                bg.append(bkgnd_)
            
            meansc = np.mean(sc)
            meanrc = np.mean(rc)
            meanbg = np.mean(bg)
            if (meansc > score_best) and (meanrc > recall_thresh):
                nu_best = n
                loggamma_best = g
                score_best = meansc
            scores.append([n, g, meansc, meanrc, meanbg])

    # train best model
    clf_best = svm.OneClassSVM(kernel='rbf', nu=nu_best,
                               gamma=10**loggamma_best)
    clf_best.fit(X_train)

    # write/save stuff
    clf_name = 'models/OC-SVM_n%.1f_logg%.1f.pkl'%(nu_best, loggamma_best)
#    with open(clf_name, 'wb') as output_:
#        pickle.dump(clf_best, output_)
    joblib.dump(clf_best, clf_name)
    best = [clf_best, nu_best, loggamma_best, score_best]
    print('best model is nu=%f, log10(gamma)=%f, score=%f'%(nu_best, loggamma_best, score_best))
    return scores, best

def classify_repo(GP, mdl_dir='models/OC-SVM_n0.1_logg-1.7.pkl'):
    # prepare data
    name_map = ['url', 'n_pyfiles', 'code_lines', 'comment_lines', 'docstring_lines',
                'test_lines','readme_lines', 'n_commits', 'commits_per_time', 'n_stars',
                'n_forks']
    pep8_map = ['E1','E2','E3','E4','E5','E7','E9','W1','W2','W3','W5','W6']
    data = {}
    for n in name_map:
        data[n] = getattr(GP, n)

    for n in pep8_map:
        data[n] = GP.pep8[n]

    data = make_features(pd.DataFrame.from_dict([data]))

    # prepare feature array
    scaler_name = 'models/scaler.pkl'
    scaler = joblib.load(scaler_name)
    minvals_name = 'models/minvals.pkl'
    minvals = joblib.load(minvals_name)
#    with open(scaler_name, "rb") as input_file:
#        scaler = pickle.load(input_file)
#    with open(minvals_name, "rb") as input_file:
#        minvals = pickle.load(input_file)
    X = np.log10(data[train_features])
    for c in X.columns:
        minval = minvals[c]
        X.loc[X[c] == -np.inf, c] = minval
        X.loc[X[c] == np.inf, c] = minval
    X = scaler.transform(X)

    # prepare model
    clf = joblib.load(mdl_dir)
#    with open(mdl_dir, "rb") as input_file:
#        clf = pickle.load(input_file)

    # generate pred
    return clf.predict(X)[0]

if __name__ == '__main__':
    # directory info
    good_dir = 'repo_data/top_stars_stats_Python.txt'
    bad_dir = 'repo_data/bottom_stars_stats_Python_local.txt'
    
    # train params
    nu = np.linspace(0.01, 1, 20)  # 0-1 range
    loggamma = np.linspace(-4, 0, 10)
    n_cv = 3

    # prepare data
    X, Xb = prepare_data(good_dir, bad_dir)
    
    # train model
    scores, best = train_model(X, Xb, nu, loggamma, n_cv)
