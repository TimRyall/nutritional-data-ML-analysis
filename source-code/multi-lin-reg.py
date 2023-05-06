import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

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

all_mse = []

from sklearn.model_selection import KFold
# define the number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)

# loop through each fold
for train_index, test_index in kf.split(X):
    # get the training and testing data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # train and fit the linear regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # apply the linear model to our test data
    y_hat = reg.predict(X_test)
    mse  = mean_squared_error(y_test, y_hat) # calculate MSE
    all_mse.append(mse) # add mse to lsit


print(f'Mean: {np.mean(all_mse)} SD: {np.sqrt(np.var(all_mse))}')



##################################################
##################################################




##################################################
##################################################