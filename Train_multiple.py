# This code is for mass training of multiple regressors to find the best
import Regression
import datetime
record = {}
start = datetime.datetime.now()
prev = start
for i in range(20, 100, 10):
    lr = Regression.LinReg(filename="Known_set_Bacillus.xlsx", test_size=i/100.0)
    record[i] = {'LinReg': lr, 'score': lr.asses_model(prtf=False), 'coefs': lr.get_coefs(prtf=False)}
    now = datetime.datetime.now()
    print('Finished iteration in time: {0}'.format(now-prev))
    prev = now
best = record[max(record, key=lambda x: record[x]['score'][0])]['LinReg']