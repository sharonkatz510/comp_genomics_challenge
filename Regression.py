import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from bunch import Bunch

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
AAs = list(set(x for x in NT2AA_dict.values() if x is not None))


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


def get_CpG_content(sequence: str):
    """Get relative amount of GC in sequence"""
    return sequence.count('GC')/float(len(sequence))


class LinReg:
    """
    Gather data from file, extract additional features and labels, later use to train a linear regressor
    """
    def __init__(self, filename=None, drop_wins=False, data=None, label=None,train_test_data = None):
        self.X = data
        self.Y = label
        if data is None:
            data = pd.read_excel(filename, index_col='Gene index')
            self.str_seriesses = data[['PA', 'UTR5', 'ORF']]
            self.X = data.drop(columns=['PA', 'UTR5', 'ORF', 'argenin frequnecy '])
            self.add_features()
            self.Y = data['PA']

        self.normalized_Y = self.Y.apply(lambda x: ((x - min(self.Y)) / (max(self.Y) - min(self.Y))) ** 0.1)
        if drop_wins: self.drop_windows()
        [self.model, self.train_test_data] = self.get_model(train_test_data)

    def add_features(self):
        orf_cols = ['{0}_freq'.format(x) for x in AAs]
        orf_aa_freqs = add_aas_freq(self.str_seriesses['ORF'], orf_cols,AAs)
        self.X = pd.concat([self.X, orf_aa_freqs], axis=1)
        self.X['free_var'] = np.ones(len(self.X))
        self.X['avg_win'] = np.mean(self.X.iloc[:, 4:104], axis=1)
        self.X['std_win'] = np.std(self.X.iloc[:, 4:104], axis=1)
        self.X['orf_CpG_content'] = self.str_seriesses['ORF'].apply(get_CpG_content)
        self.X['utr_CpG_content'] = self.str_seriesses['UTR5'].apply(get_CpG_content)
        self.X['TATA_boxes'] = self.str_seriesses['UTR5'].apply(lambda x: x.count('TATA'))
        for nuc in ['A', 'T', 'G', 'C'] : self.X["utr_{0}_freq".format(nuc)] = get_nuc_freq(nuc,
                                                                                            self.str_seriesses['UTR5'])
        for nuc in ['A', 'T', 'G', 'C'] : self.X["orf_{0}_freq".format(nuc)] = get_nuc_freq(nuc,
                                                                                            self.str_seriesses['ORF'])

    # TODO: <deprecate?>
    def get_conf_mat(self, plot_flag=False, return_flag=False):
        """
        :param plot_flag: plots if True
        :param return_flag: returns feat-feat and feat-label correlation mat\vec if True
        """
        conf_mat = pd.concat([self.X.drop(columns='free_var'), self.Y], axis=1).corr(method='spearman')
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

    # TODO: <deprecate?>
    def drop_windows(self):
        """
        Remove "window" features if irrelevant
        """
        self.X = self.X.drop(columns=[name for name in self.X.columns if 'dow' in name])

    def plot_col_num(self, col_num):
        """
        this method scatter plots the feature and label plot
        :param col_num:  data frame column number
        """
        col = self.X.iloc[:, col_num]
        plt.figure()
        plt.scatter(self.X[col], self.Y)
        plt.xlabel(col), plt.ylabel('PA')

    def get_model(self,data=None):
        """
        Train a linear regressor based on inside or outside data and return model and data split
        """
        # TODO: cross-validation (k-fold)
        if not data:
            data = Bunch()
            data.x_train, data.x_test, data.y_train, data.y_test = train_test_split(self.X, self.Y, test_size=0.1)
        reg = LinearRegression()
        reg.fit(data.x_train, data.y_train)
        return reg, data

    def asses_model(self, data=None, Training = False):
        """
        Asses model performance and print results
        Training = use training data instead of test data
        """
        reg = self.model
        x_test = self.train_test_data.x_test ; y_test = self.train_test_data.y_test
        if Training:
            x_test = self.train_test_data.x_train ; y_test = self.train_test_data.y_train
        if data:
            x_test = data.x ; y_test = data.y
        y_pred = pd.Series(reg.predict(x_test), index=y_test.index)
        print("R^2 for data is:{0}".format(reg.score(x_test, y_test)))
        print("Spearman correlation score:{0}".format(spearmanr(y_pred, y_test)))

    def visualize_model_performance(self, data=None):
        """Plot model prediction vs label"""
        x_test = self.train_test_data.x_test ; y_test = self.train_test_data.y_test
        if data:    # Allows performance comparison between models on same dataset
            x_test = data.x
            y_test = data.y
        y_pred = pd.Series(self.model.predict(x_test), index=y_test.index)
        plt.figure()
        y_pred_reind = y_pred.copy()
        y_test_reind = y_test.copy()
        y_pred_reind.index = range(len(y_pred))
        y_test_reind.index = range(len(y_test))
        asc_ind = y_test_reind.sort_values().index
        plt.scatter(range(len(y_pred)), y_pred_reind.iloc[asc_ind], label="y_pred")
        plt.scatter(range(len(y_test)), y_test_reind.iloc[asc_ind], label="y_test")
        plt.xticks(asc_ind)
        plt.legend()
        plt.title('Test PA')
        plt.show()

    def get_coefs(self, n=10, pltf=False):
        """
            return n primary model coefficients
            :pltf is plot_flag
        """
        mdl = self.model
        abscoefs = abs(mdl.coef_)
        if len(mdl.coef_)<n:
            n = len(mdl.coef_)-1
        if pltf:
            plt.figure(), plt.bar(np.arange(len(abscoefs)).ravel(), abscoefs)
            plt.title('Regression coefficients values')
            plt.xticks(range(len(self.X.columns)), self.X.columns)
        sorted_idx = np.flip(abscoefs.argsort())
        sorted_features = lr.X.columns[sorted_idx]

        for i in range(n):
            print("feature name:{0}, feature coeff:{1}".format(sorted_features[i], abscoefs[sorted_idx][i]))


if __name__ == "__main__":
    from feature_selection import *
    from math import sqrt
    lr = LinReg("Known_set_Bacillus.xlsx", drop_wins=False)
    best_features = ffs(round(sqrt(len(lr.X))), lr.X, lr.Y)
    # This part is ugly but not sure how to solve this 'cause pandas are fucking mutable TODO: find better solution
    best_data = Bunch()
    best_data.x_train = lr.train_test_data.x_train[list(best_features)].copy()
    best_data.x_test = lr.train_test_data.x_test[list(best_features)].copy()
    best_data.y_train = lr.train_test_data.y_train.copy()
    best_data.y_test = lr.train_test_data.y_test.copy()
    # End of ugly part
    only_best = LinReg(data=lr.X[list(best_features)], label=lr.Y, train_test_data=best_data)
    only_best.visualize_model_performance()
    print('***All features results***')
    print('--training results--')
    lr.asses_model(Training=True)
    print('--test results--')
    lr.asses_model()
    print('***Only best features results***')
    print('--training results--')
    only_best.asses_model(Training=True)
    print('--test results--')
    only_best.asses_model()


    # only_best.get_coefs(n=15, pltf=True)
    # lr.coef(n=15, pltf=True)
    # self = lr

    # plt.imshow(lr.X[:-2].corr() - np.eye(len(lr.X.columns)))
    # flc, ffc = lr.get_conf_mat(return_flag=True)
    # lr.get_conf_mat(plot_flag=True)
    # plt.figure(), plt.bar(range(len(flc)), flc), plt.xticks(range(len(flc)), list(lr.X.columns), size=7, rotation=45)