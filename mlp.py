from gb import X_train
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mglearn

X,y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, random_state = 42)
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_Train_scaled = (X_train-mean_on_train)/std_on_train
X_Test_scaled = (X_test-mean_on_train)/std_on_train
mlp = MLPClassifier(max_iter=5000, solver='adam', random_state=0, alpha=0.1, hidden_layer_sizes=[120,120]).fit(X_Train_scaled,y_train)
print(f'Accuracy on training set: {mlp.score(X_Train_scaled, y_train)}')
print(f'Accuracy on test set: {mlp.score(X_Test_scaled, y_test)}')
mglearn.plots.plot_2d_separator(mlp,X_Train_scaled,fill=True,alpha=0.3)
mglearn.discrete_scatter(X_Train_scaled[:,0], X_Train_scaled[:,1],y_train)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.show()


