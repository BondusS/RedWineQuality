import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Dataset = pd.read_csv("winequality-red.csv")
Dataset.info()
ColumnsToDrop = ['fixed acidity', 'residual sugar', 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH']
Dataset = Dataset.drop(ColumnsToDrop, axis=1)
sns.heatmap(Dataset.corr(),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(xs=Dataset['quality'].to_numpy(),
           ys=Dataset['alcohol'].to_numpy(),
           zs=Dataset['volatile acidity'].to_numpy())
plt.show()
