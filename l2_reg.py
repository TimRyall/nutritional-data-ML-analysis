import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#####################################################################
# import data
df = pd.read_csv("report/Rel_2_Nutrient_file.csv")

# remove the 4 columns mentioned in 'Data set information' Section
df = df.drop(columns=df.columns[:3]) # removing: Key, Classifcation, Name

# read in the core nutrient file
core_nutrients_df = pd.read_csv("report/Core_Nutrient_details.csv")
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

# Assign 70% training data 30% testing data for each data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a polynomial feature transformer with degree 2
transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train = transformer.fit_transform(X_train) # adujst X to add polynomial terms
X_test = transformer.fit_transform(X_test) 

lambdas = np.logspace(-4, 4, num=20)
test_mse_per_lambda = []
train_mse_per_lambda = []
for lambda_l1 in lambdas:
    test_current_lambda_mse = []
    train_current_lambda_mse = []
    for trial in range(1):
        # train and fit the linear regression model
        reg = Ridge(alpha=lambda_l1, max_iter=10000) # alpha is our regularisation param
        reg.fit(X_train, y_train)

        # apply the linear model to our test data
        y_hat_test = reg.predict(X_test)
        y_hat_train = reg.predict(X_train)

        test_mse = mean_squared_error(y_test, y_hat_test) # calculate MSE
        train_mse = mean_squared_error(y_train, y_hat_train)

        test_current_lambda_mse.append(test_mse) # add mse to lsit
        train_current_lambda_mse.append(train_mse)
    #print(f'lambda: {lambda_l1}: {np.mean(current_lambda_mse)}') # average MSE
    test_mse_per_lambda.append(np.mean(test_current_lambda_mse))
    train_mse_per_lambda.append(np.mean(train_current_lambda_mse))


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))
ax1.loglog(lambdas, test_mse_per_lambda, color='lightcoral', label='Testing Error')
ax1.loglog(lambdas, train_mse_per_lambda, color='darkseagreen', label='Training Error')
ax1.set_title('Training and testing error for model \n using L2 regularisation')
ax1.set_xlabel('Regularisation parameter (lambda) \n Model complexity increases ->')
ax1.set_ylabel('Error (MSE)')
ax1.legend()
ax1.invert_xaxis()

g_gap = result = [b - a for a, b in zip(train_mse_per_lambda, test_mse_per_lambda)]

ax2.loglog(lambdas, g_gap, color='gray')
ax2.set_title(f'Generalisation Gap')
ax2.set_xlabel(f'Regularisation parameter (lambda) \n Model complexity increases ->')
ax2.set_ylabel('Generalisation Gap (Etrain - Etest)')
plt.xlim(min(lambdas)-1, max(lambdas)+1)
plt.ylim(min(g_gap)-1, max(g_gap)+1)

for i in len(g_gap):
    if g_gap[i] == min(g_gap):
        print(f'GG: {g_gap[i]} LAMB: {lambdas[i]}, TESTE: {test_mse_per_lambda[i]}')

ax2.invert_xaxis()



#plt.savefig(f'{input_feature}.png', bbox_inches='tight')
plt.show()




##################################################
##################################################




##################################################
##################################################