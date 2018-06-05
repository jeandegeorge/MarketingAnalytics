import pandas as pd
import numpy as np

# Soumission 2 avec variables indicatives (X=1 si le contact fait une donation au mois X)


### Format data 1

df = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/final_try_do.csv')
df_id = df[['contact_id', 'donation', 'calibration', 'amount', 'act_date']].drop_duplicates(subset=['contact_id', 'donation', 'calibration', 'amount', 'act_date'], keep="first")
df_id = df_id.set_index('contact_id')

dummies = pd.get_dummies(df['y_m'])

df.drop(['Unnamed: 0', 'y_m'], axis=1, inplace=True)

df = pd.concat([df['contact_id'], dummies], axis=1)

cols = [cols for cols in df.columns if cols not in ['contact_id', 'NA_NA']]

df = df.groupby('contact_id')[cols].sum()

df = pd.concat([df_id, df], axis=1)


### Format data 2

df2 = pd.read_csv('/Users/Jean/Desktop/Marketing Analytics/devoir2/final_try_pa.csv')
df_id = df2[['contact_id', 'donation', 'calibration', 'amount', 'act_date']].drop_duplicates(subset=['contact_id', 'donation', 'calibration', 'amount', 'act_date'], keep="first")
df_id = df_id.set_index('contact_id')

dummies = pd.get_dummies(df2['y_m'])

df2.drop(['Unnamed: 0', 'y_m'], axis=1, inplace=True)

df2 = pd.concat([df2['contact_id'], dummies], axis=1)

cols = [cols for cols in df2.columns if cols not in ['contact_id', 'NA_NA']]

df2 = df2.groupby('contact_id')[cols].sum()

df2 = pd.concat([df_id, df2], axis=1)

df['contact_id'] = df.index
df2['contact_id'] = df2.index

df = df.merge(df2, left_on=['contact_id', 'calibration', 'donation', 'amount', 'act_date'], right_on=['contact_id', 'calibration', 'donation', 'amount', 'act_date'], how='outer')

print(df.columns[df.isnull().any()].tolist())


### Format to apply model

y = np.array(df.loc[df['calibration'] == 1, 'donation'])
y = y.astype(int)

X = np.array(df.loc[df['calibration'] == 1,
                    [str(c) for c in df.columns if c not in ['contact_id', 'calibration', 'donation', 'amount', 'act_date']]])
X = X.astype(float)

X_test = np.array(df.loc[df['calibration'] == 0,
                         [str(c) for c in df.columns if c not in ['contact_id', 'calibration', 'donation', 'amount', 'act_date']]])
X_test = X_test.astype(float)


### Feature Selection (select k best variables according to F-test criterion)

from sklearn.feature_selection import SelectKBest, chi2, f_classif

feature_num = 4 #np.shape(X)[1]
kbest = SelectKBest(chi2, k=feature_num).fit(X, y)

X_new = kbest.transform(X)
X_test_new = kbest.transform(X_test)


### Apply Naive Bayes algorithm with cross-validation

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)

from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()
naive.fit(X_new, y)
print(accuracy_score(y, naive.predict(X_new)))
print(compute_score(naive, X_new, y, scoring='accuracy'))

print(sum(naive.predict(X_test_new)))

### Format submission table

contact_id = np.array(df.loc[df['calibration'] == 0, 'contact_id'])

X_test_new = kbest.transform(X_test)
donation = naive.predict(X_test_new)
sum(donation)

pred = pd.DataFrame(contact_id)
pred['donation'] = donation

df = df.set_index('contact_id')

### Write txt file

pred.to_csv('/Users/Jean/Desktop/result3.txt', header=None, index=None, sep='\t')


