import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import PCA


train_features = ['code/files', 'comment/code', 'test/code', 'readme/code',
                  'docstring/code', 'E1/code', 'E2/code', 'E3/code',
                  'E4/code', 'E5/code', 'E7/code', 'W1/code', 'W2/code',
                  'W3/code', 'W6/code', 'code_lines']


def make_features(df, filter_bottom=False):
    """
    Processing and Feature Engineering.
    """
    df['code/files'] = df['code_lines'] / df['n_pyfiles']
    df['comment/code'] = df['comment_lines'] / df['code_lines']
    df['test/code'] = df['test_lines'] / df['code_lines']
    df['readme/code'] = df['readme_lines'] / df['code_lines']
    df['docstring/code'] = df['docstring_lines'] / df['code_lines']
    for p in ['E1', 'E2', 'E3', 'E4', 'E5', 'E7',
              'E9', 'W1', 'W2', 'W3', 'W5', 'W6']:
        df['%s/code' % p] = df[p] / df['code_lines']

    df = df.dropna(how='any').drop_duplicates()
    if filter_bottom is True:
        for f in ['code/files', 'comment/code', 'test/code',
                  'readme/code', 'docstring/code']:
            df = df[df[f] > 0]

    return df


def load_data(dir, filter_bottom=False):
    """
    Load data from text file.
    """
    fields = ['url', 'n_pyfiles', 'code_lines', 'comment_lines',
              'docstring_lines', 'test_lines', 'readme_lines', 'n_commits',
              'commits_per_time', 'n_stars', 'n_forks', 'E1', 'E2',
              'E3', 'E4', 'E5', 'E7', 'E9', 'W1', 'W2', 'W3', 'W5', 'W6']
    df = pd.read_csv(dir, names=fields)
    df = make_features(df, filter_bottom)
    return df


def prepare_data(good_dir, bad_dir):
    """
    Preprocess, take log, fill in missing values, standardize.
    """
    df_good = load_data(good_dir, filter_bottom=True)
    df_bad = load_data(bad_dir)

    # log data, really useful feature
    X = np.log10(df_good[train_features])
    Xb = np.log10(df_bad[train_features])

    # replace log10(0) values with (min val - 1), i.e. order of mag. lower
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
    scaler.fit(pd.concat([X, Xb], axis=0))  # scale over X+Xb data together!
    X_s = scaler.transform(X)   # scaled
    Xb_s = scaler.transform(Xb)  # scaled
    scaler_name = 'models/scaler.pkl'
    minvals_name = 'models/minvals.pkl'
    joblib.dump(scaler, scaler_name)
    joblib.dump(minvals, minvals_name)

    # save as arrays
    np.save('models/X.npy', X_s)
    np.save('models/Xb.npy', Xb_s)
    return X_s, Xb_s, X, Xb


def focal_score(y_pred_test, y_pred_bkgnd, nu, gamma):
    """
    metric: Try to maximize recall whilst including as few background
    samples as possible. Ref: W. S. Lee and B. Liu, 'Learning with positive
    and unlabeled examples using weighted logistic regression'
    """
    # recall
    recall = len(np.where(y_pred_test == 1)[0]) / float(len(y_pred_test))

    # fraction of background samples with a positive classification
    bckgnd_focal_frac = len(np.where(y_pred_bkgnd == 1)[0]) / float(len(y_pred_bkgnd))

    try:
        score = recall**2 / bckgnd_focal_frac
    except ZeroDivisionError:
        print(("recall=%f, background_focal_frac=%f,"
               "nu=%f, gamma=%f") % (recall, bckgnd_focal_frac, nu, gamma))
        score = 0
    return score, recall, bckgnd_focal_frac


def random_train_test_split(X, train_frac=0.8):
    """
    Randomly shuffle the data, split into Test/Train.
    Useful for Cross Validation. 
    """
    N = len(X)
    rN = np.arange(0, N)
    np.random.shuffle(rN)  # randomly shuffle data
    train_i = rN[0: int(train_frac * N)]
    test_i = rN[int(train_frac * N):]

    X_train, X_test = X[train_i], X[test_i]
    return X_train, X_test


def train_model(X_s, Xb_s, X, Xb, nu, loggamma, n_cv=3, recall_thresh=0.85):
    """
    Train model using "focal_score()" metric, subject to
    recall > recall_thresh constraint.
    """
    # iterate over hypers, cv
    scores = []
    nu_best = 0
    loggamma_best = 0
    score_best = 0
    for n in nu:
        for g in loggamma:
            sc, rc, bg = [], [], []
            for i in range(n_cv):
                X_train, X_test = random_train_test_split(X_s)
                Xb_train, Xb_test = random_train_test_split(Xb_s)

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

    # write positive/negative classes to file
    y_X = clf_best.predict(X_s)
    y_Xb = clf_best.predict(Xb_s)
    X_pos = np.concatenate((X[y_X == 1], Xb[y_Xb == 1]))  # unscaled
    X_neg = np.concatenate((X[y_X == -1], Xb[y_Xb == -1]))  # unscaled
    np.save('models/X_pos_unscaled.npy', X_pos)
    np.save('models/X_neg_unscaled.npy', X_neg)

    # write/save stuff
    clf_name = 'models/OC-SVM.pkl'
    joblib.dump(clf_best, clf_name)
    best = [clf_best, nu_best, loggamma_best, score_best]
    print('best model is nu=%f, log10(gamma)=%f, score=%f' % (nu_best, loggamma_best, score_best))
    return scores, best


def get_PCs(X_s, Xb_s, plot=False):
    """
    Get and plot Principal Components (PCs) for Positive and Background data.
    """
    pca = PCA(n_components=2)
    pca.fit(np.concatenate((X_s, Xb_s)))
    pca_name = 'models/pca.pkl'
    joblib.dump(pca, pca_name)
    X_PC = pca.transform(X_s)
    Xb_PC = pca.transform(Xb_s)

    # plot
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(X_PC[:, 0], X_PC[:, 1], '.', label='200+ stars')
        plt.plot(Xb_PC[:, 0], Xb_PC[:, 1], '.', label='0 stars/forks', alpha=0.6)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        #plt.title('explained variance: %.2f' % np.sum(pca.explained_variance_ratio_))
        plt.legend()
        plt.savefig('images/PCs.png')
    
    return X_PC, Xb_PC


def classify_repo(GP, mdl_file='models/OC-SVM.pkl'):
    """
    Predict class of scraped repo using pre-trained One-Class SVM model.
    """

    # prepare data
    name_map = ['url', 'n_pyfiles', 'code_lines', 'comment_lines',
                'docstring_lines', 'test_lines', 'readme_lines',
                'n_commits', 'commits_per_time', 'n_stars', 'n_forks']
    pep8_map = ['E1', 'E2', 'E3', 'E4', 'E5', 'E7', 'E9', 'W1', 'W2',
                'W3', 'W5', 'W6']
    data = {}
    for n in name_map:
        data[n] = getattr(GP, n)
    for n in pep8_map:
        data[n] = GP.pep8[n]
    data = make_features(pd.DataFrame.from_dict([data]))

    # prepare features, preprocess.
    scaler_name = 'models/scaler.pkl'
    minvals_name = 'models/minvals.pkl'
    scaler = joblib.load(scaler_name)
    minvals = joblib.load(minvals_name)
    X = np.log10(data[train_features])
    for c in X.columns:
        minval = minvals[c]
        X.loc[X[c] == -np.inf, c] = minval
        X.loc[X[c] == np.inf, c] = minval

    # prepare model
    clf = joblib.load(mdl_file)
    repo_pred = clf.predict(scaler.transform(X))[0]

    # generate pred, X is kept unscaled for plotting!
    return repo_pred, X.values


if __name__ == '__main__':
    # directory info
    good_dir = 'repo_data/top_stars_stats_Python.txt'
    bad_dir = 'repo_data/bottom_stars_stats_Python.txt'

    # train params
    nu = np.linspace(0.01, 1, 20)  # 0-1 range
    loggamma = np.linspace(-4, 0, 10)
    n_cv = 3

    # prepare data
    X_s, Xb_s, X, Xb = prepare_data(good_dir, bad_dir)

    # calculate PCs if desired
    X_PC, Xb_PC = get_PCs(X_s, Xb_s)

    # train model
    scores, best = train_model(X_s, Xb_s, X, Xb, nu, loggamma, n_cv)
