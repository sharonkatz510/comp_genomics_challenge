import pandas as pd
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_excel("Known_set_Bacillus.xlsx", index_col='Gene index') # optiona: add [, index_col='Gene index'] to the DF parameters
n_samp = 20
print(data.index)
data.head(n_samp)


y = data["PA"]
x = data[sort(list(set(data.columns)-{"PA","UTR5","ORF"}))]

xs = x[[col for col in x.columns if col.find("dow") < 0]]

x.head(n_samp)





# cmx = pd.concat([xs,y],axis = 1).corr()
# tcmx = np.abs(cmx.values-np.eye(cmx.shape[0]))

# print(f"Correlation matrix:\n\n{tcmx}\n\n")
# print(np.max(tcmx[:,-1]))

# imshow(tcmx), xticks(np.arange(cmx.shape[0]),cmx.columns), yticks(np.arange(cmx.shape[0]),cmx.columns), colorbar(), plt.show()


# y.head(n_samp)
# list(y)


cm = pd.concat([x, y], axis=1).corr()
tcm = np.abs(cm.values-np.eye(cm.shape[0]))

print(f"Max Label Correlation: {np.max(tcm[-1,:])} \n")
print(f"Label Correlation: \n{tcm[-1,:]} \n\n")


imshow(tcm)
xticks(np.arange(cm.shape[0]), cm.columns, size=7, rotation=45)
yticks(np.arange(cm.shape[0]), cm.columns, size=7, rotation=45)
colorbar()
plt.show()


X = x.copy()
X["Free_var"] = np.ones(x.shape[0])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = LinearRegression().fit(x_train, y_train)
print("R^2 for linear regression is:", reg.score(x_test, y_test))


if __name__ == "__main__":
    pass

