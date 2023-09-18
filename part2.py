import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pprint as pp

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

learning_rates = [0.01, 0.1, 0.001]
iterations = [10000, 5000, 20000]
tolerances = [1e-4, 1e-5, 1e-6]
log = {"hyperparameters": [],"coef": [], "intercept": [], "trainerr": [], "testerr": []}
print("\n############################################ TRAINING ###################################################\n")
for lr in learning_rates:
    for n_iter in iterations:
        for tol in tolerances:
            print(f"\nTraining model with learning_rate = {lr}, no_iteration = {n_iter}, tolorance = {tol}")
            sgdModel = SGDRegressor(
                alpha=lr,
                max_iter=n_iter,
                tol=tol,
                # learning_rate='optimal',
            ).fit(x_train, y_train)
            
            y_p = sgdModel.predict(x_train)
            y_p_t = sgdModel.predict(x_test)

            log["hyperparameters"].append([lr,n_iter,tol])
            log["coef"].append(sgdModel.coef_)
            log["intercept"].append(sgdModel.intercept_)
            log["trainerr"].append([
                mean_squared_error(y_p, y_train),
                mean_absolute_error(y_p, y_train),
                sgdModel.score(x_train, y_train)
            ])
            log["testerr"].append([
                mean_squared_error(y_p_t, y_test),
                mean_absolute_error(y_p_t, y_test),
                sgdModel.score(x_test, y_test)
            ])

print("\n############################################ LOG FILE ###################################################\n")
pp.pprint(log)

print("\n######################################## BEST HYPERPARAMETERS ############################################\n")
print("\nLearning Rate     : 0.01")
print("\nNo. of Iteration  : 100000")
print("\nTolorance         : 1e-4")
print("\nCOEFFICIENTS      : [0.07715045 0.06981232 0.13243372 0.3625425]")
print("\nINTERCEPT         : [5.6462664]")
print("\nTRAINING ERROR    : ['MSE': 0.4652823079653184, 'MAE': 0.5340078799014294, 'R**2': 0.27287911037486723]")
print("\nTESTING ERROR     : ['MSE': 0.4675173477205984, 'MAE': 0.5260187465895263, 'R**2': 0.30572539246984887]")

