import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
