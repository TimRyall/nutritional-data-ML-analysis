import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Data for the bars
x = ['Simple linear', 'Multiple Linear', 'Simple Polynomial', 'Multiple Polynomial', 'Default SVR', 'Tuned SVR', 'Polynomial L1 Reg', 'Polynomial L2 Reg']
y = [124510.10, 18925.41, 118422.08, 7028.20, 11879.86, 8492.39, 6125.52, 6375.50]
errors = [20471.11, 4852.92, 37505.93, 2196.84, 3388.29, 3729.37, 989.26, 2287.14]

# Create a bar chart
plt.bar(x, y, yerr=errors, capsize=5, ecolor='grey', color='darkseagreen')
plt.yscale('log')
for i in range(len(x)):
    plt.text(i, y[i] + errors[i] + 0.5, y[i] , ha='center', va='bottom')

# Add a title and labels for the axes
plt.title('MSE Comparision of our diffrent regession models \n (with +-1 sd error bars)')
plt.xlabel('Model used')
plt.ylabel('Average MSE')

# Rotate the x-axis labels
plt.xticks(rotation=45)


# Adjust the margins to prevent the plot from being cut off
plt.savefig('kfold_reg_comparison.png', bbox_inches='tight')
plt.show()
