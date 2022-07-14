# This module takes care of all utility functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

NT2AA_dict = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': None, 'TAG': None,
    'TGC': 'C', 'TGT': 'C', 'TGA': None, 'TGG': 'W',
}


def get_codons(seq):
    return [seq[i:i + 3] for i in range(0, int(len(seq)), 3)]


def get_aa(seq):
    """"
    Get DNA sequence and transcribe to amino acids
    """
    start = seq.index('ATG')    # seek start codon
    if len(seq[start:]) % 3 > 0:
        seq = seq[start:-1 - (len(seq[start:]) % 3)]
    codons = [seq[i:i + 3] for i in range(0, int(len(seq)), 3)] #split string to list of 3 letter codons
    aas = [NT2AA_dict.get(codon) for codon in codons] # translate
    if None in aas: # seek end codon
        return "".join(aas[:aas.index(None)])
    else:
        return "".join(aas)


def get_aa_freq(series, wanted_aa):
    """
    returns the input AA's frequency
    """
    aas = series.apply(get_aa)
    return aas.apply(lambda x: len(x) and x.count(wanted_aa) / len(x))


def add_aas_freq(data, col_names, aas):
    """"
    return AA frequencies in data
    """
    x = pd.DataFrame(columns = col_names)
    for cn, aa in zip(col_names, aas): x[cn] = get_aa_freq(data,aa)
    return x


def get_nuc_freq(wanted_nuc, data):
    return data.apply(lambda x: len(x) and x.count(wanted_nuc)/len(x))


def get_conf_mat(X, Y, plot_flag=False, return_flag=False):
    """
    :param plot_flag: plots if True
    :param return_flag: returns feat-feat and feat-label correlation mat\vec if True
    """
    conf_mat = pd.concat([X.drop(columns='free_var'), Y], axis=1).corr(method='spearman')
    truncated_conf_mat = np.abs(conf_mat.values - np.eye(conf_mat.shape[0]))
    feat_label_corr = truncated_conf_mat[-1, :]
    feature_feature_corr = truncated_conf_mat[0:-2, 0:-2]
    print("\nMax Feature-Label Correlation: {0} \n".format(np.max(truncated_conf_mat[-1, :])))
    if plot_flag:
        print("Feature-Label Correlation: \n{0} \n".format(feat_label_corr))
        plt.imshow(truncated_conf_mat, interpolation='nearest', )
        plt.xticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
        plt.yticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
        plt.colorbar(), plt.show()
    if return_flag: return feat_label_corr, feature_feature_corr


@np.vectorize
def get_tata_loc(x: str):
    """ Return distance of tata-box from string end or -1 if no tata box"""
    if x.find('TATA') == -1:
        return -1
    else:
        return len(x) - x.find('TATA')


@np.vectorize
def gc_count(x: str):
    """Return length of longest gc"""
    return x.count('GC')


@np.vectorize
def num_start_codons(x: str):
    return x.count('ATG')


def spear_corr(x, y):
    return pd.Series(x).corr(pd.Series(y), method="spearman")


def sfs(k, data, label, fn=spear_corr, coef_limit=None, test_size=0.2):
    """
    This function implements ffs using multi-feature regression and evaluation
    function fn
    k: number of features,
    data: pandas DataFrame,
    label: pandas series
    fn: Feature evaluation function
    test_size: proportion of test data from total data
    """
    data["free_var"] = np.ones(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size)
    best_features = 'free_var',
    max_feature_score = 0
    for i in range(k):
        available_features = list(data.columns.drop(labels=list(best_features)))
        feature_scores = {}
        for feature_name in available_features:
            feature_list = list(best_features)
            feature_list.append(feature_name)
            feature_comb = x_train[feature_list]
            reg = LinearRegression().fit(feature_comb, y_train)
            feature_scores[feature_name] = fn(reg.predict(x_test[feature_list]), y_test)
        if max(feature_scores.values()) < max_feature_score:
            break
        else:
            best_features = best_features + (max(feature_scores, key=feature_scores.get),)
            max_feature_score = max(feature_scores.values())

    if coef_limit:
        feature_list = list(best_features)
        feature_comb = x_train[feature_list]
        reg = LinearRegression().fit(feature_comb, y_train)
        good_features = [ind[0] for ind, coef in np.ndenumerate(reg.coef_) if abs(coef) > coef_limit]
        best_features = [item for i, item in enumerate(feature_list) if i in good_features]

    return best_features


def sbs(k, data, label, fn=spear_corr, decrease_limit=0.8, test_size=0.2):
    """
    This function implements fbs using multi-feature regression and evaluation
    function fn
    k: number of features,
    data: pandas DataFrame,
    label: pandas series
    fn: Feature evaluation function
    test_size: proportion of test data from total data
    """
    data["free_var"] = np.ones(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size)
    best_features = tuple(data.columns)
    reg = LinearRegression().fit(x_train, y_train)
    best_score = fn(reg.predict(x_test), y_test)
    for i in range(len(best_features), k, -1):
        available_features = list(best_features)
        feature_scores = {}
        for feature_name in (k for k in available_features if k is not 'free_var'):
            feature_list = list(best_features)
            feature_list.remove(feature_name)
            feature_comb = x_train[feature_list]
            reg = LinearRegression().fit(feature_comb, y_train)
            feature_scores[feature_name] = fn(reg.predict(x_test[feature_list]), y_test)
        if max(feature_scores.values())< decrease_limit*best_score:
            break
        else:
            best_features = list(best_features)
            best_features.remove(min(feature_scores, key=feature_scores.get))
            best_features = tuple(best_features)
    return best_features
