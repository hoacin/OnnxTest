#Inspired by https://towardsdatascience.com/deploy-sci-kit-learn-models-in-net-core-applications-90e24e572f64

from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

dataset = datasets.fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.1)

model_parameters = {
    'n_estimators': 500,
    'max_depth': 6,
    'min_samples_split': 5,
    'learning_rate': 0.01,
    'loss': 'ls'
}

model = ensemble.GradientBoostingRegressor(**model_parameters)
model.fit(X_train, y_train)

testVector = [[2.5,5,5,3,4.5,5,3.5,-2]]
testResult = model.predict(testVector)*100000
print(testResult)

#y_predictions = model.predict(X_test)
#rmse = mean_squared_error(y_test,y_predictions, squared=True)

initial_type = [('float_input', FloatTensorType([None, 8]))]
onnx = convert_sklearn(model, initial_types=initial_type)
with open("artifacts/california_housing.onnx", "wb") as f:
    f.write(onnx.SerializeToString())

