import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import pickle

from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['TARGET']= boston.target

X = df.drop('TARGET',axis=1)
y = df['TARGET']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(X)
### VIF SELECTION

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF']=[variance_inflation_factor(scaled,i) for i in range (scaled.shape[1])]
vif['Columns']= X.columns
# print(vif)

X.drop(['RAD','TAX'],axis=1,inplace=True)
print(X.shape)
scaler1 = StandardScaler()
scaled1 = scaler1.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
print(X_train.shape),print(X_test.shape)
# linear regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_test,y_test)
print(regression.score(X_train,y_train))
#pickle.dump(regression,open('linear_regression','wb'))
# model = pickle.load(open('linear_regression','rb'))
# print('linear model: ',model.predict(scaler1.transform([[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,15.3,396.90,4.98]])))


# Regularization

from  sklearn.linear_model import LassoCV,Lasso,RidgeCV,Ridge,ElasticNetCV,ElasticNet

lassocv = LassoCV(alphas=None,cv=10,normalize=True)
lassocv.fit(X_train,y_train)
print('lasso_best_alpha_value: ',lassocv.alpha_)

lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(X_train,y_train)

pickle.dump(lasso,open('linear_regression.pickle','wb'))
model = pickle.load(open('linear_regression.pickle','rb'))
print(model.predict([[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,15.3,396.90,4.98]]))


def adj_r2(x,y,models):
    r2 = models.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return 'R2 : ',r2 ,'Adj_R2 : ',adjusted_r2

print(adj_r2(X_train,y_train,regression))
print(adj_r2(X_train,y_train,lasso))
#print('Lasso: ',lasso.predict(scaler1.transform([[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,15.3,396.90,4.98]])))
#alphas = np.random.uniform(low=0, high=10, size=(50,))
#ridgecv= RidgeCV(alphas=alphas,cv=10,normalize=True)
#ridgecv.fit(X_train,y_train)

#print(ridgecv.alpha_)

#ridge = Ridge(alpha=ridgecv.alpha_)
#ridge.fit(X_train,y_train)
#print('Ridge',ridge.predict(scaler1.transform([[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,15.3,396.90,4.98]])))

#df2=pd.DataFrame({'Actual':y_test, 'predicted':ridge.predict(X_test)})
#print(df2.head())


#elasticNetcv = ElasticNetCV(alphas=None,cv=10)
#elasticNetcv.fit(X_train,y_train)
#print(elasticNetcv.alpha_)

# elasticNet = ElasticNet(alpha=elasticNetcv.alpha_)
#elasticNet.fit(X_train,y_train)
#print(lasso.score(X_test,y_test))
#print(ridge.score(X_test,y_test))
#print(elasticNet.score(X_test, y_test))

pd.options.display.max_columns
print(X.head())




