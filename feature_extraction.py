
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, f1_score

from sklearn.metrics import confusion_matrix



# Data Preparation
#According to the slide we will be working with dataset that focuses on company D and the following sports:
#football, gold and tennis.

df = pd.read_csv('recruitmentdataset-2022-1.3.csv')
# we chose company D
df = df.loc[df['company'] == 'D']

# converting categorical and ordinal value to numerical values (boolean values)
bl_features = ['ind-debateclub', 'ind-programming_exp', 'ind-international_exp',
                'ind-entrepeneur_exp', 'ind-exact_study','decision']

for i in bl_features:
    df[i].replace(True, 1, inplace = True)
    df[i].replace(False, 0, inplace = True)

#converting gender column into numerical values by making new column for each gender

df['female'] = 0
df['male'] = 0
df['other'] = 0

for i in df.index:
  if df['gender'][i] == 'female':
    df.at[i, 'female'] = 1
    df.at[i, 'male'] = 0
    df.at[i, 'other'] = 0

  elif df['gender'][i] == 'male':
    df.at[i, 'female'] = 0
    df.at[i, 'male'] = 1
    df.at[i, 'other'] = 0

  if df['gender'][i] == 'other':
    df.at[i, 'female'] = 0
    df.at[i, 'male'] = 0
    df.at[i, 'other'] = 1
X = df.drop(columns=['decision', 'Id'])
y = df['decision']

# converting nationality column to numerical values
df['Dutch'] = 0
df['German'] = 0
df['Belgian'] = 0

for i in df.index:
    if df['nationality'][i] == 'Dutch':
        df.at[i, 'Dutch'] = 1
        df.at[i, 'German'] = 0
        df.at[i, 'Belgian'] = 0

    elif df['nationality'][i] == 'German':
        df.at[i, 'Dutch'] = 0
        df.at[i, 'German'] = 1
        df.at[i, 'Belgian'] = 0

    else:
        df.at[i, 'Dutch'] = 0
        df.at[i, 'German'] = 0
        df.at[i, 'Belgian'] = 1

#converting degree column into numerical values

df['bachelor'] = 0
df['master'] = 0
df['phd'] = 0

for i in df.index:

    if df['ind-degree'][i] == 'bachelor':
        df.at[i, 'bachelor'] = 1
        df.at[i, 'master'] = 0
        df.at[i, 'phd'] = 0

    elif df['ind-degree'][i] == 'master':
        df.at[i, 'bachelor'] = 0
        df.at[i, 'master'] = 1
        df.at[i, 'phd'] = 0

    else:
        df.at[i, 'bachelor'] = 0
        df.at[i, 'master'] = 0
        df.at[i, 'phd'] = 1

df.head()

#adding the three sports columns and deleting the unncessary rows
df['football'] = 0
df['golf'] = 0
df['tennis'] = 0

for i in df.index:

    if df['sport'][i] == 'Football':
        df.at[i, 'football'] = 1
        df.at[i, 'golf'] = 0
        df.at[i, 'tennis'] = 0

    elif df['sport'][i] == 'Golf':
        df.at[i, 'football'] = 0
        df.at[i, 'golf'] = 1
        df.at[i, 'tennis'] = 0

    elif df['sport'][i] == 'Tennis':
        df.at[i, 'football'] = 0
        df.at[i, 'golf'] = 0
        df.at[i, 'tennis'] = 1

    else:
        df.drop(axis = 0, index = i, inplace = True)


#removing unecessary columns

df.drop('nationality', axis = 1, inplace = True)
df.drop('ind-degree', axis = 1, inplace = True)
df.drop('company', axis = 1, inplace = True)
df.drop('sport',axis = 1, inplace = True)
df.drop('gender',axis = 1, inplace = True)

X = df.drop(columns=['decision', 'Id'])
y = df['decision']

k = 5
selector = SelectKBest(score_func=mutual_info_classif, k=k)

selector.fit(X, y)

X_new = selector.transform(X)

selected_feature_indices = selector.get_support(indices=True)

selected_features = X.columns[selected_feature_indices]

feature_scores_df = pd.DataFrame({'feature_name': X.columns, 'information_gain_score': selector.scores_})
feature_scores_df = feature_scores_df.sort_values(by='information_gain_score', ascending=False)

print(feature_scores_df)


# Data Preparation
X = df[['ind-university_grade', 'ind-exact_study', 'ind-programming_exp', 'age','ind-languages']]
y = df['decision']
# Dynamic feature selection based on mutual information gain
k = 5
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

# Display the top selected features based on mutual information gain
feature_scores_df = pd.DataFrame({'feature_name': X.columns, 'information_gain_score': selector.scores_})
feature_scores_df = feature_scores_df.sort_values(by='information_gain_score', ascending=False)
print(feature_scores_df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.1, stratify=y, random_state=42)

# Hyperparameter tuning using GridSearchCV with Stratified K-Fold cross-validation
k_values = [3, 5, 7, 9, 11]
weights_options = ['uniform', 'distance']
distance_metrics = ['euclidean', 'manhattan']

param_grid = {
    'n_neighbors': k_values,
    'weights': weights_options,
    'metric': distance_metrics
}
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
best_f1 = 0
best_hyperparams = None
best_test_size = None

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=test_size, stratify=y, random_state=42)
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_hyperparams = grid_search.best_params_
        best_test_size = test_size

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluation on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"\nPrecision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}")

print(f"Best Test Size: {best_test_size}")
print(f"Best Hyperparameters: {best_hyperparams}")
print(f"Best F1-Score: {best_f1:.2f}")




# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the figure as a PNG image
plt.savefig("/mnt/c/User/aldej/Desktop/confusion_matrix.png", bbox_inches='tight', dpi=300)

