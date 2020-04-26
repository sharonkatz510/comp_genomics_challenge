import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }


def get_aa(seq):
    if len(seq) % 3 > 0: seq = seq[:-1 - (len(seq) % 3)]
    codons = [seq[i:i + 3] for i in range(0, int(len(seq)), 3)]
    return "".join([NT2AA_dict[codon] for codon in codons])


class LinReg:
    def __init__(self, filename, drop_wins=False):
        data = pd.read_excel(filename, index_col='Gene index')
        self.str_seriesses = data[[col for col in data.columns if col in ['PA', 'UTR5', 'ORF']]]
        x = data[[col for col in data.columns if col not in ['PA', 'UTR5', 'ORF']]]
        self.X = x.drop(columns='argenin frequnecy ')
        self.add_features(len(x))
        if drop_wins: self.drop_windows()
        self.Y = data['PA']

    def add_features(self, lx):
        self.add_aa_freq()
        self.X['free_var'] = np.ones(lx)

    def add_aa_freq(self):
        self.X['Arg_freq'] = self.get_aa_freq('R')
        self.X['Ala_freq'] = self.get_aa_freq('A')
        self.X['Gly_freq'] = self.get_aa_freq('G')
        self.X['Val_freq'] = self.get_aa_freq('V')

    def get_aa_freq(self, wanted_aa):
        aas = [get_aa(nt) for nt in self.str_seriesses['ORF'].tolist()]
        p = [aa.count(wanted_aa) / len(aa) for aa in aas]
        return p

    def get_conf_mat(self, plot_flag=False, return_flag=False):
        """
        :param plot_flag: plots if True
        :param return_flag: returns feat-feat and feat-label correlation mat\vec if True
        """
        conf_mat = pd.concat([self.X.drop(columns='free_var'), self.Y], axis=1).corr(method='spearman')
        truncated_conf_mat = np.abs(conf_mat.values - np.eye(conf_mat.shape[0]))
        feat_label_corr = truncated_conf_mat[-1, :]
        feature_feature_corr = truncated_conf_mat[0:-2, 0:-2]
        print(f"\nMax Feature-Label Correlation: {np.max(truncated_conf_mat[-1, :])} \n")
        if plot_flag:
            print(f"Feature-Label Correlation: \n{feat_label_corr} \n")
            plt.imshow(truncated_conf_mat, interpolation='nearest', )
            plt.xticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
            plt.yticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
            plt.colorbar(), plt.show()
        if return_flag: return feat_label_corr, feature_feature_corr

    def drop_windows(self):
        wins = [name for name in self.X.columns if 'dow' in name]
        self.X = self.X.drop(columns=wins)

    def plot_col_num(self, col_num):
        """
        :param col_num:  data frame column number
        """
        plt.figure()
        col = self.X.columns[col_num]
        plt.scatter(self.X[col], lr.Y)
        plt.xlabel(col)
        plt.ylabel('PA')

    def get_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2)
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        print("R^2 for linear regression is:", reg.score(x_test, y_test))
        return reg, reg.score(x_test, y_test)


if __name__ == "__main__":
    lr = LinReg("Known_set_Bacillus.xlsx", drop_wins=False)
    # self = lr
    # flc, ffc = lr.get_conf_mat(return_flag=True)
    # lr.get_conf_mat(plot_flag=True)
    # plt.figure(), plt.bar(range(len(flc)), flc), plt.xticks(range(len(flc)), list(lr.X.columns), size=7, rotation=45)
    # lr.get_model()
