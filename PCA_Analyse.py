import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Dataset = pd.read_csv("winequality-red.csv")
sns.heatmap(Dataset.corr(),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()
X_df = Dataset.drop(['quality'], axis=1)
X_np = X_df.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
pca = PCA()
x_pca = pca.fit_transform(X_st)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()
# Компонент выбран на основе графика
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(X_st)
