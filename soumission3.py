import pandas as pd
import numpy as np

# la meilleure, dernière soumission

# Importation des données générées via SQL (mêmes queries que pour la soumission1)

recent2 = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/recent2.csv')
recent2.drop('Unnamed: 0', axis=1, inplace=True)

july2 = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/july2.csv')
july2.drop('Unnamed: 0', axis=1, inplace=True)

pa2 = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/pa2.csv')
pa2.drop('Unnamed: 0', axis=1, inplace=True)

resprate2 = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/resprate2.csv')
resprate2.drop('Unnamed: 0', axis=1, inplace=True)

df = recent2.merge(july2, left_on='contact_id', right_on='contact_id', how='outer')
df = df.merge(pa2, left_on='contact_id', right_on='contact_id', how='outer')
df = df.merge(resprate2, left_on='contact_id', right_on='contact_id', how='outer')

df.avg1.fillna(0, inplace=True)
df.sd1.fillna(0, inplace=True)

df_save = df[['contact_id', 'sd_july16', 'sd1', 'sd_pa']]
df.drop(['sd_july16', 'sd1', 'sd_pa'], axis=1, inplace=True)

print(df.columns[df.isnull().any()].tolist())


########## Format to apply classifier

y = np.array(df.loc[df['calibration'] == 1, 'donation'])
y = y.astype(int)

X = np.array(df.loc[df['calibration'] == 1,
                    [str(c) for c in df.columns if c not in ['contact_id', 'calibration', 'donation', 'amount', 'act_date']]])
X = X.astype(float)

X_test = np.array(df.loc[df['calibration'] == 0,
                         [str(c) for c in df.columns if c not in ['contact_id', 'calibration', 'donation', 'amount', 'act_date']]])
X_test = X_test.astype(float)


### Feature Selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif

feature_num = np.shape(X)[1]
kbest = SelectKBest(chi2, k=feature_num).fit(X, y)

X_new = kbest.transform(X)
X_test_new = kbest.transform(X_test)

### Classifier

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)

# Tune parameters through a grid search
# turn run to True to run grid search; False to model with outcome of grid search

from sklearn.model_selection import GridSearchCV, StratifiedKFold

run = False

if run:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }

    forest = RandomForestClassifier()

    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(estimator=forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(X, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'bootstrap': True, 'max_depth': 6, 'max_features': 'auto', 'min_samples_leaf': 3,
                  'min_samples_split': 2, 'n_estimators': 50}

    model = RandomForestClassifier(**parameters)
    model.fit(X, y)

print(compute_score(model, X, y, scoring='accuracy'))

print(sum(model.predict(X_test)))



### Regressor; format

amount = np.array(df.loc[df['donation'] == 1, 'amount'])
amount = amount.astype(int)

df = df.merge(df_save, left_on='contact_id', right_on='contact_id', how='outer')

X2 = np.array(df.loc[df['donation'] == 1, ['avg1', 'avg_pa', 'avg_july16', 'sd1', 'sd_july16', 'sd_pa']])
X2 = X2.astype(float)

### Try three different regressors: Random Forest, Lasso, Linear. Get cross validation scores (R2)

### RF regressor (best)

for i in range(1, np.shape(X2)[1]+1):
    feature_num = i
    kbest = SelectKBest(chi2, k=feature_num).fit(X2, amount)

    X2_new = kbest.transform(X2)

    from sklearn.ensemble import RandomForestRegressor

    RF_reg = RandomForestRegressor()
    RF_reg.fit(X2_new, amount)
    print(str(i)+' '+str(np.mean(cross_val_score(RF_reg, X2_new, amount, scoring='r2', cv=5))))

### Try linear regression

for i in range(1, np.shape(X2)[1]+1):
    feature_num = i
    kbest = SelectKBest(chi2, k=feature_num).fit(X2, amount)

    X2_new = kbest.transform(X2)

    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()
    lin.fit(X2_new, amount)
    print(str(i) + ' ' + str(np.mean(cross_val_score(lin, X2_new, amount, scoring='r2', cv=5))))

### Try Lasso

from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import Lasso
lasso = Lasso()


for i in range(1, np.shape(X2)[1]+1):

    feature_num = i
    kbest = SelectKBest(chi2, k=feature_num).fit(X2, amount)
    X2_new = kbest.transform(X2)

    lasso.fit(X2_new, amount)
    print(str(i) + ' ' + str(np.mean(cross_val_score(lasso, X2_new, amount, scoring='r2', cv=5))))


### Format tab separated file for submission

contact_id = np.array(df.loc[df['calibration'] == 0, 'contact_id'])

donation_p = model.predict_proba(X_test)

donation = [1-donation_p[i][0] for i in range(np.shape(donation_p)[0])]

pred = pd.DataFrame(contact_id)
pred['donation'] = donation

df2 = df.loc[df['calibration'] == 0, ['contact_id', 'avg1', 'avg_pa', 'avg_july16', 'sd1', 'sd_july16', 'sd_pa']]
df2['donation'] = donation

X2_test = np.array(df.loc[df['calibration'] == 0, ['avg1', 'avg_pa', 'avg_july16', 'sd1', 'sd_july16', 'sd_pa']])
X2_test = X2_test.astype(float)
RF_reg.fit(X2, amount)
donation_amt = RF_reg.predict(X2_test)

df2['amount'] = donation_amt

df2['score'] = df2['amount']*df2['donation']

# expected value must be above 2

df2['reach_out'] = np.where(df2['score'] > 2, 1, 0)


# save results
import csv
df2[['contact_id', 'reach_out']].to_csv('/Users/Jean/Desktop/result5.txt', header=None, index=None, sep='\t')

