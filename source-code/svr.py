import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


#####################################################################
# import data
df = pd.read_csv("source-code/Rel_2_Nutrient_file.csv")

# remove the 4 columns mentioned in 'Data set information' Section
df = df.drop(columns=df.columns[:3]) # removing: Key, Classifcation, Name

# read in the core nutrient file
core_nutrients_df = pd.read_csv("source-code/Core_Nutrient_details.csv")
# only use the component column 
core_nutrients = core_nutrients_df['Component']

# make a list of the features that are core nutrients
columns_to_keep = []
for col in df.columns:
    for nutrient in core_nutrients:
        if nutrient in col:
            columns_to_keep.append(col)
            break

# drop the features that are not core nutrients 
df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

# rename Energy column
df = df.rename(columns={'Energy with dietary fibre, equated \r\n(kJ)': 'Energy \n(kJ)'})

# when a feature has > 50% NaN or zero values we remove it
# count the number of non-null, non-zero values in each column
obs_counts = df.apply(lambda x: x[x.notnull() & (x != 0)].count())
# define the threshold value for the minimum number of observations
min_obs = df.shape[0] * 0.5 # number of observations * 50%
# filter the columns based on the minimum number of observations
selected_columns = obs_counts[obs_counts >= min_obs].index
# update dataframe with only the selected columns
df = df[selected_columns]


# replace all NaN values with assumed 0
df = df.fillna(0)
# replace commas with nothing in all columns
df = df.replace(',', '', regex=True)
# remove new lines from feature names for readability
df = df.rename(columns=lambda x: x.replace('\n', ''))
df = df.rename(columns=lambda x: x.replace('\r', ''))
# convert all columns to numeric type
df = df.apply(pd.to_numeric, errors='coerce')


# calculate the correlation between Energy and every other column
corr_with_energy = (df.corrwith(df['Energy (kJ)']).abs())
# define the threshold value
threshold = 0.3
# filter the correlation coefficients to remove elements below the threshold
corr_with_energy = corr_with_energy[corr_with_energy >= threshold]

df = df[corr_with_energy.index]












df = df.sample(frac=1) # randomly shuffle data
X = df.drop(columns=df.columns[0]) # observations for inputs (drop target)
y = df['Energy (kJ)'] # observations for our output







# Scale the data to standarise inputs
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn import svm # support vector machines

best_mse = 1000000000

kernals = ['linear', 'poly', 'rbf', 'sigmoid']
reg_params = np.logspace(-1, 1, num=10)
epsilons = np.logspace(-1, 1, num=10)

for kernal in kernals:
    for reg_param in reg_params:
        for epsilon in epsilons:
            all_mse = []
            for _ in range(5):
                # split data in to training and testing
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                # create and train SVM model
                model = svm.SVR(kernel=kernal, C=reg_param, epsilon=epsilon)
                model.fit(X_train, y_train)

                # predict the test data
                y_hat = model.predict(X_test)

                # report the MSE of the model
                mse  = mean_squared_error(y_test, y_hat) # calculate MSE
                all_mse.append(mse)
            # if new combination finds better MSE update current best
            if (np.mean(all_mse) < best_mse):
                best_combination = [kernal, reg_param, epsilon]
                best_mse = np.mean(all_mse)

print("MSE:", best_mse, "COMBIN:", best_combination)