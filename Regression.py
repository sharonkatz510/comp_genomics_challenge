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


def get_aa(codons):
    return "".join([NT2AA_dict[codon] for codon in codons])


class LinReg:
    def __init__(self, filename):
        data = pd.read_excel(filename, index_col='Gene index')
        self.str_seriesses = data[[col for col in data.columns if col in ['PA', 'UTR5', 'ORF']]]
        self.X = data[[col for col in data.columns if col not in ['PA', 'UTR5', 'ORF']]]
        self.X["Free_var"] = np.ones(len(self.X))
        self.Y = data['PA']

    def get_conf_mat(self, plot_flag=False):
        conf_mat = pd.concat([self.X, self.Y], axis=1).corr()
        truncated_conf_mat = conf_mat.values - np.eye(conf_mat.shape[0])
        print(f"\nMax Feature-Label Correlation: {np.max(truncated_conf_mat[-1, :])} \n")
        print(f"Feature-Label Correlation: \n{truncated_conf_mat[-1, :]} \n")
        if plot_flag:
            plt.imshow(truncated_conf_mat)
            plt.xticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
            plt.yticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
            plt.colorbar(), plt.show()

    def data_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2)
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        print("R^2 for linear regression is:", reg.score(x_test, y_test))


if __name__ == "__main__":
    lr = LinReg("Known_set_Bacillus.xlsx")
    # lr.get_conf_mat(True)
    # lr.data_model()
