import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pprint as pp
import matplotlib.pyplot as plt


print("\n########################################## PREPROCESSING #################################################\n")
print("[1] Reading the Dataset")
df = pd.read_csv("https://raw.githubusercontent.com/divyam-prajapati/ML_Assignment_LR_GD/main/winequality-red.csv", sep=";")

print("[2] Preprocesing the Dataset")
df.dropna()
df.drop_duplicates()
# df.info()

print("[3] Spliting the Dataset in features and target varibles")
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

print("[4] Feature selection")
x = x.drop(["volatile acidity", "residual sugar",	"chlorides",	"free sulfur dioxide",	"total sulfur dioxide",	"density",	"pH"], axis=1)
x.describe()

print("[5] Scaling the features")
scaler = StandardScaler()
x[x.columns] = scaler.fit_transform(x[x.columns])
x.describe()

y.describe()

print("[6] Spliting the Dataset in train test for both x & y")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
# print(len(x_train), len(x_test), len(y_train), len(y_test))

print("[7] Converting inputs to np arrays")
X_train = np.array([
    x_train['fixed acidity'].values.tolist(),
    x_train['citric acid'].values.tolist(),
    x_train['sulphates'].values.tolist(),
    x_train['alcohol'].values.tolist()
  ])
X_test = np.array([
    x_test['fixed acidity'].values.tolist(),
    x_test['citric acid'].values.tolist(),
    x_test['sulphates'].values.tolist(),
    x_test['alcohol'].values.tolist()
  ])
Y_train = y_train.to_numpy()
Y_test = y_test.to_numpy()

X_train = np.insert(X_train, 0, 1, axis=0)
X_test = np.insert(X_test, 0, 1, axis=0)

# MAIN FUNCTION FOR LINEAR REGRESSION
class linear_regression():
 
  def __init__(self, learning_rate, no_iteration, tolorance):
    self.learning_rate = learning_rate
    self.no_iteration = no_iteration
    self.tolorance = tolorance

  def gradient_descent(self, x, y, w):
    # res = w[0] +  w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] * x[3] - y
    # print("Eq: ", res.mean(), (res * x[0]).mean(), (res * x[1]).mean(), (res * x[2]).mean(), (res * x[3]).mean())  # .mean() is a method of np.ndarray
    # print("Vec: ", (x.dot(x.T.dot(w) - y))/len(y))
    # print("X:     ", x)
    # print("RES:   ", x.T.dot(w) - y)
    # print("X*RES: ", x.dot(x.T.dot(w) - y))
    # print("mean:  ", (x.dot(x.T.dot(w) - y))/len(y))
    return (x.dot(x.T.dot(w) - y))/len(y)

  # def gradient_descent(x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06):
  def fit(self, x, y, w=[0.0, 0.0, 0.0, 0.0, 0.0]):
    vector = np.array(w)
    err_his = {'MSE':[],'MAE':[],'R2':[]}
    for i in range(self.no_iteration):
      diff = -self.learning_rate * np.array(self.gradient_descent(x, y, vector))
      if np.all(np.abs(diff) <= self.tolorance):
        break
      vector += diff
      e = self.error(x, y, vector)
      err_his["MSE"].append(e[0])
      err_his["MAE"].append(e[1])
      err_his["R2"].append(e[2])
      # print("EPOCH", i ," MSE: ", e[0],"     MAE: ", e[1],"     R2: ", e[2])
    # print(i)
    return vector, err_his

  def predict(self, w, x):
    return w.T.dot(x)
    # return w[0] + w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] * x[3]

  def error(self, x, y, w):
    yp = self.predict(w, x)
    return [mean_squared_error(y, yp), mean_absolute_error(y, yp), r2_score(y, yp)]
  

learning_rates = [0.01, 0.1, 0.001]
iterations = [10000, 5000, 20000]
tolerances = [1e-4, 1e-5, 1e-6]
log = {"hyperparameters": [],"weights": [], "trainerr": [], "testerr": []}
print("\n############################################ TRAINING ###################################################\n")
for lr in learning_rates:
    for n_iter in iterations:
        for tol in tolerances:
            print(f"\nTraining model with learning_rate = {lr}, no_iteration = {n_iter}, tolorance = {tol}")
            model = linear_regression(
                learning_rate=lr,
                no_iteration=n_iter,
                tolorance=tol
            )
            weights, his = model.fit(
                X_train,
                Y_train,
            )
            log["hyperparameters"].append([lr,n_iter,tol])
            log["weights"].append(weights)
            log["trainerr"].append(model.error(X_train, Y_train, weights))
            log["testerr"].append(model.error(X_test, Y_test, weights))
            # print(f"Final weights:", log["weights"][-1])
            # print(f"Training Error:", log["trainerr"][-1])
            # print(f"Testing Error:", log["testerr"][-1])
            # print("\n====================================================================================================\n")

print("\n############################################ LOG FILE ###################################################\n")
pp.pprint(log)

print("\n######################################## BEST HYPERPARAMETERS ############################################\n")
print("\nLearning Rate     : 0.01")
print("\nNo. of Iteration  : 100000")
print("\nTolorance         : 1e-4")
print("\nWEIGHTS           : [5.60755747 0.03350934 0.08643086 0.12718095 0.38175852]")
print("\nTRAINING ERROR    : ['MSE': 0.4889196807512065, 'MAE': 0.5268462746603718, 'R**2': 0.290933869220359]")
print("\nTESTING ERROR     : ['MSE': 0.4702540945119479, 'MAE': 0.5403431986961312, 'R**2': 0.2828459677989706]")

print("\n################################### PLOTS FOR BEST HYPERPARAMETERS #######################################\n")

model = linear_regression(learning_rate=0.01,no_iteration=100000,tolorance=1e-4)
weights, his = model.fit(X_train,Y_train)

plt.plot(his['MSE'])
plt.title('Mean Square Errors vs No. of Iterations')
plt.xlabel('No. of Iterations')
plt.show()

plt.plot(his['MAE'])
plt.title('Mean Absolute Errors vs No. of Iterations')
plt.xlabel('No. of Iterations')
plt.show()

plt.plot(his['R2'])
plt.title('R**2 vs No. of Iterations')
plt.xlabel('No. of Iterations')
plt.show()

plt.plot(his['MSE'])
plt.plot(his['MAE'])
plt.plot(his['R2'])
plt.title('Errors vs No. of Iterations')
plt.xlabel('No. of Iterations')
plt.legend(['MSE', 'MAE', 'R2'], loc='upper right')
plt.show()

plt.bar(['Fixed Acidity', 'Citric Acid', 'Sulphates', 'Alcohol'], [0.04374161, 0.07417663, 0.12954444, 0.38544707], width = 0.3, color='blue')
plt.title('Weights for Linear Regression Features')
plt.xlabel('Features')
plt.ylabel('Weight Coeficients')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()
