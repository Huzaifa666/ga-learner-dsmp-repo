# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)
X = data.drop(['customer.id','paid.back.loan'], axis = 1)
y = data['paid.back.loan']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, test_size = 0.3)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
fully_paid.hist()

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate'] = [x.replace("%","") for x in X_train['int.rate']]
X_train['int.rate'] = [float(x) for x in X_train['int.rate']]

X_test['int.rate'] = [x.replace("%","") for x in X_test['int.rate']]
X_test['int.rate'] = [float(x) for x in X_test['int.rate']]

X_train['int.rate'] = X_train['int.rate']/100
X_test['int.rate'] = X_test['int.rate']/100

num_df = X_train.select_dtypes(include = 'number')
cat_df = X_train.select_dtypes(exclude = 'number')
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = num_df.columns
fig, axes = plt.subplots(nrows = 9, ncols = 1, figsize = (10,50))
for i in range(0,9):
    sns.boxplot(x=y_train, y = num_df[cols[i]], ax=axes[i])

# Code ends here


# --------------
# Code starts here
cols = cat_df.columns
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14,10))
col = []
for i in range(0,2):
    for j in range(0,2):
        col = cols[i * 2 + j]
        axes[i][j] = sns.countplot(x=X_train[col], hue = y_train, ax=axes[i,j])


# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Code starts here
for i in cat_df:
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.fit_transform(X_test[i])

for i in y_train:
    if i == 0:
        i.replace(0, "No", inplace = True)
    if i == 1:
        i.replace(1, "Yes", inplace = True)

for i in y_test:
    if i == 0:
        i.replace(0, "No", inplace = True)
    if i == 1:
        i.replace(1, "Yes", inplace = True)

model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model2 = DecisionTreeClassifier(random_state = 0)
p_tree = GridSearchCV(estimator = model2, param_grid = parameter_grid, cv = 5)
p_tree.fit(X_train, y_train)
y_pred2 = p_tree.predict(X_test)
acc_2 = accuracy_score(y_test, y_pred2)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(p_tree.best_estimator_, out_file=None,
                                feature_names=X.columns, filled = True,  
                                class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)
# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


