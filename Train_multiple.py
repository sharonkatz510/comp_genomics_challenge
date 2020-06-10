# This code is for mass training of multiple regressors to find the best
import Regression
import datetime
from utils import sbs

# record = {}
# start = datetime.datetime.now()

# for idx, i in enumerate(range(100)):
#     lr = Regression.LinReg(filename="Known_set_Bacillus.xlsx", test_size=0.5, drop_wins=True, normalize=True)
#     record[i] = {'model': lr, 'score': lr.asses_model(prtf=False)}
#     now = datetime.datetime.now()
#     print('Finished {0} iterations in time: {1}'.format(idx+1, now-start))

# best = record[max(record, key=lambda x: record[x]['score'][0])]['model']
lr = Regression.LinReg(filename="Known_set_Bacillus.xlsx", test_size=0.5, drop_wins=True, normalize=True)
best_features = sbs(10, lr.X, lr.Y, decrease_limit=0.9, test_size=0.5)
only_best = Regression.LinReg(data=lr.X[list(best_features)], label=lr.Y, test_size=0.5)
