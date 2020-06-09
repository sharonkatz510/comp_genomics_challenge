from utils import *
from sklearn.utils import Bunch
from scipy.stats import spearmanr


class LinReg:
    """
    Gather data from file, extract additional features and labels, later use to train a linear regressor
    """

    def __init__(self, **kwargs):
        filename = kwargs.get('filename', None)
        drop_wins = kwargs.get('drop_wins', False)
        data = kwargs.get('data', None)
        label = kwargs.get('label', None)

        self.X = data
        self.Y = label
        if data is None:
            tmp_data = pd.read_excel(filename, index_col='Gene index')
            self.str_seriesses = tmp_data[['PA', 'UTR5', 'ORF']]
            self.X = tmp_data.drop(columns=['PA', 'UTR5', 'ORF', 'argenin frequnecy '])
            self.add_features()
            self.Y = tmp_data['PA']

        self.normalized_Y = self.Y.apply(lambda x: ((x - min(self.Y)) / (max(self.Y) - min(self.Y))) ** 0.1)
        if drop_wins: self.X = self.X.drop(columns=[name for name in self.X.columns if 'dow' in name])
        [self.model, self.train_test_data] = self.get_model(**kwargs)

    def add_features(self):
        orf_cols = ['orf_Arg_freq', 'orf_Ala_freq', 'orf_Gly_freq', 'orf_Val_freq']
        orf_aa_freqs = add_aas_freq(self.str_seriesses['ORF'], orf_cols, ['R', 'A', 'G', 'V'])
        self.X = pd.concat([self.X, orf_aa_freqs], axis=1)
        self.X['free_var'] = np.ones(len(self.X))
        self.X['avg_win'] = np.mean(self.X.iloc[:, 4:104], axis=1)
        self.X['std_win'] = np.std(self.X.iloc[:, 4:104], axis=1)
        self.X['TATA_loc'] = self.str_seriesses['UTR5'].apply(get_tata_loc)
        self.X['GC_count'] = self.str_seriesses['UTR5'].apply(gc_count)
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

    def get_model(self, **kwargs):
        """
        Train a linear regressor based on inside or outside data and return model and data split
        """
        # TODO: cross-validation (k-fold)
        train_test_data = kwargs.get('train_test_data', None)
        test_size = kwargs.get('test_size', 0.3)
        normalize = kwargs.get('normalize', False)
        if train_test_data is None:
            train_test_data = Bunch()
            y = self.normalized_Y if normalize else self.Y
            train_test_data.x_train, train_test_data.x_test, \
            train_test_data.y_train, train_test_data.y_test = \
                train_test_split(self.X, y, test_size=test_size)
        reg = LinearRegression()
        reg.fit(train_test_data.x_train, train_test_data.y_train)
        return reg, train_test_data

    def asses_model(self, data=None, Training=False,prtf = True):
        """
        Asses model performance and print results
        Training = use training data instead of test data
        """
        reg = self.model
        x_test = self.train_test_data.x_test
        y_test = self.train_test_data.y_test
        if Training:
            x_test = self.train_test_data.x_train
            y_test = self.train_test_data.y_train
        if data:
            x_test = data.x
            y_test = data.y
        y_pred = pd.Series(reg.predict(x_test), index=y_test.index)
        if prtf:
            print("R^2 for data is:{0}".format(reg.score(x_test, y_test)))
            print(spearmanr(y_pred, y_test))
        return spearmanr(y_pred, y_test)

    def visualize_model_performance(self, data=None):
        """Plot model prediction vs label"""
        x_test = self.train_test_data.x_test
        y_test = self.train_test_data.y_test
        if data:  # Allows performance comparison between models on same dataset
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

    def get_coefs(self, n=10, pltf=False, prtf = True):
        """
            return n primary model coefficients
            :pltf is plot_flag
        """
        mdl = self.model
        abscoefs = abs(mdl.coef_)
        if len(mdl.coef_) < n:
            n = len(mdl.coef_) - 1
        if pltf:
            plt.figure(), plt.bar(np.arange(len(abscoefs)).ravel(), abscoefs)
            plt.title('Regression coefficients values')
            plt.xticks(range(len(self.X.columns)), self.X.columns)
        sorted_idx = np.flip(abscoefs.argsort())
        sorted_features = self.X.columns[sorted_idx]
        if prtf:
            for i in range(n):
                print("feature name:{0}, feature coeff:{1}".format(sorted_features[i], abscoefs[sorted_idx][i]))
        return mdl.coef_


if __name__ == "__main__":
    lr = LinReg(filename="Known_set_Bacillus.xlsx", drop_wins=True)
    best_features = ffs(round(len(lr.X)**0.5), lr.X, lr.Y)

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
