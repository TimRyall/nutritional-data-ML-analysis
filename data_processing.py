import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from sklearn.preprocessing import StandardScaler

#####################################################################

abbreviated_labels = ['Energy', 
                        'Moisture',
                        'Fat, total',
                        'Carbs w/ sugar alc',
                        'Magnesium',
                        'Alpha tocopherol',
                        'Vitamin E',
                        'Sat fatty acids',
                        'Monounsat fatty acids',
                        'Ployunsat fatty acids',
                        'Trans fatty acids']

def corelation_map(features):
    # calculate the correlation between columns
    corr_matrix = features.corr(numeric_only = True).abs()

    # plot the correlation matrix as a heatmap
    heatmap = sns.heatmap(corr_matrix.round(2), cmap="Greens", annot=True)
    plt.title('Correlation matrix of features in model')
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')

    heatmap.set_xticklabels(abbreviated_labels)
    heatmap.set_yticklabels(abbreviated_labels)
    plt.savefig('corr_matix.png', bbox_inches='tight')#plt.savefig('rest.png', bbox_inches='tight')
    # display the heatmap
    plt.show()

##########################################
def energy_corelation():
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


    print(df.columns)
    # calculate the correlation between Energy and every other column
    corr_with_energy = (df.corrwith(df['Energy (kJ)']).abs())
    # define the threshold value
    threshold = 0.3
    # filter the correlation coefficients to remove elements below the threshold
    corr_with_energy = corr_with_energy[corr_with_energy >= threshold]

    print(corr_with_energy)

    # plot the correlation coefficients as a bar chart
    plot = corr_with_energy.plot(kind='bar', color='darkseagreen')
    plt.title('Correlation between Energy and other Features \n(where correlation > 0.3)')
    plt.xlabel('Feature name')
    plt.ylabel('Absolute correlation with Energy')
    plot.set_xticklabels(abbreviated_labels)
    # Adjust the margins to prevent the plot from being cut off
    plt.savefig('corr_energy_others.png', bbox_inches='tight')
    plt.show()


    # plot the heat map
    corelation_map(df[corr_with_energy.index])






##################################################
##################################################




##################################################
##################################################