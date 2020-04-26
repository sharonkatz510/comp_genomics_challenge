import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_excel("Known_set_Bacillus.xlsx", index_col='Gene index')
# n_samp = 20
# data.head(n_samp)

y = data["PA"]
x = data[[col for col in data.columns if col not in ["PA", "UTR5", "ORF"]]]
# x.head(n_samp)

# cmx = pd.concat([xs,y],axis = 1).corr()
# tcmx = np.abs(cmx.values-np.eye(cmx.shape[0]))

# print(f"Correlation matrix:\n\n{tcmx}\n\n")
# print(np.max(tcmx[:,-1]))

# imshow(tcmx), xticks(np.arange(cmx.shape[0]),cmx.columns), yticks(np.arange(cmx.shape[0]),cmx.columns), colorbar(), plt.show()


# y.head(n_samp)
# list(y)


conf_mat = pd.concat([x, y], axis=1).corr()
# truncated_conf_mat = np.abs(conf_mat.values - np.eye(conf_mat.shape[0]))
truncated_conf_mat = conf_mat.values - np.eye(conf_mat.shape[0])
print(f"\nMax Feature-Label Correlation: {np.max(truncated_conf_mat[-1, :])} \n")
print(f"Feature-Label Correlation: \n{truncated_conf_mat[-1, :]} \n")


plt.imshow(truncated_conf_mat)
plt.xticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
plt.yticks(np.arange(conf_mat.shape[0]), conf_mat.columns, size=7, rotation=45)
plt.colorbar()
plt.show()


X = x.copy()
X["Free_var"] = np.ones(x.shape[0])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = LinearRegression()


if __name__ == "__main__":
    reg.fit(x_train, y_train)
    print("R^2 for linear regression is:", reg.score(x_test, y_test))
