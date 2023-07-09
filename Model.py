import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

Dataset = pd.read_csv("winequality-red.csv")
X_df = Dataset.drop(['quality'], axis=1)
X_np = X_df.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
pca = PCA()
x_pca = pca.fit_transform(X_st)
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(X_st)
reviews = []
for i in Dataset['quality']:
    if 1 <= i <= 3:
        reviews.append('Bad')
    elif 4 <= i <= 7:
        reviews.append('Normal')
    elif 8 <= i <= 10:
        reviews.append('Good')
Dataset['Reviews'] = reviews
Y_df = Dataset['Reviews']
Y_np = Y_df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x_new, Y_np, stratify=Dataset['quality'])
Model = SVC(kernel='rbf').fit(x_train, y_train)
print('Score правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Score правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('Подборка лучших параметров...')
params = {'gamma': [n for n in range(1, 11, 1)],
          'C': [i*0.1 for i in range(1, 21, 1)]}
grid = GridSearchCV(estimator=Model, param_grid=params)
grid.fit(x_train, y_train)
print('Наилучший score при подборке наиболее подходящих параметров ', grid.best_score_)
print('Best gamma', grid.best_estimator_.gamma)
print('Best C', grid.best_estimator_.C)
BestModel = SVC(gamma=grid.best_estimator_.gamma,
                C=grid.best_estimator_.C,
                kernel='rbf').fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', BestModel.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', BestModel.score(x_test, y_test))
