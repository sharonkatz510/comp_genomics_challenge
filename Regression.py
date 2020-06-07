import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


class LinReg:
    """
    Gather data from file, extract additional features and labels, later use to train a linear regressor
    """

    def __init__(self, filename=None, drop_wins=False, data=None, label=None):
        self.X = data
        self.Y = label
        if data is None:
            data = pd.read_excel(filename, index_col='Gene index')
            self.str_seriesses = data[['PA', 'UTR5', 'ORF']]
            self.X = data.drop(columns=['PA', 'UTR5', 'ORF', 'argenin frequnecy '])
            self.add_features()
            self.Y = data['PA']

        self.normalized_Y = self.Y.apply(lambda x: ((x - min(self.Y)) / (max(self.Y) - min(self.Y))) ** 0.1)
        if drop_wins: self.X = self.X.drop(columns=[name for name in self.X.columns if 'dow' in name])

    def add_features(self):
        orf_cols = ['orf_Arg_freq', 'orf_Ala_freq', 'orf_Gly_freq', 'orf_Val_freq']
        orf_aa_freqs = add_aas_freq(self.str_seriesses['ORF'], orf_cols, ['R', 'A', 'G', 'V'])
        self.X = pd.concat([self.X, orf_aa_freqs], axis=1)
        self.X['free_var'] = np.ones(len(self.X))
        self.X['avg_win'] = np.mean(self.X.iloc[:, 4:104], axis=1)
        self.X['std_win'] = np.std(self.X.iloc[:, 4:104], axis=1)
        for nuc in ['A', 'T', 'G', 'C']: self.X["utr_{0}_freq".format(nuc)] = get_nuc_freq(nuc,
                                                                                           self.str_seriesses['UTR5'])
        for nuc in ['A', 'T', 'G', 'C']: self.X["orf_{0}_freq".format(nuc)] = get_nuc_freq(nuc,
                                                                                           self.str_seriesses['ORF'])

    def plot_col_num(self, col_num):
        """
        this method scatter plots the feature and label plot
        :param col_num:  data frame column number
        """
        col = self.X.iloc[:, col_num]
        plt.figure()
        plt.scatter(self.X[col], self.Y)
        plt.xlabel(col), plt.ylabel('PA')

    def get_model(self, prtf=False, pltf=False):
        """
        Train a linear regressor based on self data
        :prtf is print_flag
        :rf is return_flag
        """
        # TODO: cross-validation (k-fold)
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.normalized_Y, test_size=0.2)
        reg = LinearRegression()
        # reg = ElasticNet()
        reg.fit(x_train, y_train)

        # TODO: fix model assesment
        y_pred = pd.Series(reg.predict(x_test), index=y_test.index)
        print(spearmanr(y_pred, y_test))

        # scatter plotting y_pred : y_test
        if pltf:
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
            plt.show()

        if prtf:
            print("R^2 for train data is:{0}".format(reg.score(x_train, y_train)))
            print("R^2 for test data is:{0}".format(reg.score(x_test, y_test)))
            print("Spearman correlation score:{0}".format(spearmanr(y_pred, y_test)))

        return reg, reg.score(x_test, y_test)

    def coef(self, n=10, pltf=False):
        """
            work with model coefficients
            :pltf is plot_flag
        """
        mdl, scr = self.get_model(pltf=True)  # prtf=True
        abscoefs = abs(mdl.coef_)
        if pltf:
            plt.figure(), plt.plot(abscoefs)
        sorted_idx = np.flip(abscoefs.argsort())
        sorted_features = lr.X.columns[sorted_idx]
        for i in range(n):
            print("feature name:{0}, feature coeff:{1}".format(sorted_features[i], abscoefs[sorted_idx][i]))


if __name__ == "__main__":
    from feature_selection import *
    from math import sqrt

    lr = LinReg("Known_set_Bacillus.xlsx", drop_wins=False)

    best_features = ffs(round(sqrt(len(lr.X))), lr.X, lr.normalized_Y)
    only_best = LinReg(data=lr.X[list(best_features)], label=lr.Y)

    mdl, mdl_score = only_best.get_model(pltf=True)
    only_best.coef(n=15, pltf=True)

