# PROPERATI DATA 
# STEP WISE PRE-PROCESSING
# ML ALGORITHM APPLICATION

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.style as style
style.use('fivethirtyeight')


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

# Data Scaler
from sklearn.preprocessing import StandardScaler

# Regression
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# Set file path and viewing
data = pd.read_csv('properati.csv')
pd.set_option('display.max_columns',999)
data.head()


# delete fields not used
del data['created_on']
del data['image_thumbnail']


# Change data type
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['rooms'] = pd.to_numeric(data['rooms'], errors='coerce')
data['price_usd_per_m2'] = pd.to_numeric(data['price_usd_per_m2'], errors='coerce')
data['price_per_m2'] = pd.to_numeric(data['price_per_m2'], errors='coerce')
data['surface_covered_in_m2'] = pd.to_numeric(data['surface_covered_in_m2'], errors='coerce')
data['surface_total_in_m2'] = pd.to_numeric(data['surface_total_in_m2'], errors='coerce')

# Remove duplicates
sum(data.duplicated(data.columns))
data = data.drop_duplicates(data.columns, keep='last')
sum(data.duplicated(data.columns))


# Count and plot
variables = data.columns

count = []

for variable in variables:
		length = data[variable].count()
		count.append(length)
	
count_pct = np.round(100 * pd.Series(count) / len(data), 2)
count_pct

plt.figure(figsize=(10,6))
plt.barh(variables, count_pct)
plt.title('Count of available data in percent', fontsize=15)
plt.show()



# Delete non null values and plot
data = data[data['price'].notnull()]
len(data)
data.describe()
plt.figure(figsize=(15,6))
sns.boxplot(x='price', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of price')
plt.show()
plt.figure(figsize=(15,6))
sns.distplot(data['price'])

plt.title('Histogram of SALE PRICE')
plt.ylabel('Normed Frequency')
plt.show()


# Proportions for plotting
x = data[['price']].sort_values(by='price').reset_index()
x['PROPERTY PROPORTION'] = 1
x['PROPERTY PROPORTION'] = x['PROPERTY PROPORTION'].cumsum()
x['PROPERTY PROPORTION'] = 100* x['PROPERTY PROPORTION'] / len(x['PROPERTY PROPORTION'])
plt.plot(x['PROPERTY PROPORTION'],x['price'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to Price')
plt.xlabel('Percentage of Properties in ascending order of Price')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
data = data[(data['price'] > 1) & (data['price'] < 50000000)]
len(data)


#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x='price', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(data['price'])
plt.title('Histogram of SALE PRICE in USD')
plt.ylabel('Normed Frequency')
plt.show()


plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(data['price'])
plt.title('Histogram of SALE PRICE in USD')
plt.ylabel('Normed Frequency')
plt.show()

plt.figure(figsize=(15,6))

#Get the data and format it
x = data[['price']].sort_values(by='price').reset_index()
x['PROPERTY PROPORTION'] = 1
x['PROPERTY PROPORTION'] = x['PROPERTY PROPORTION'].cumsum()
x['PROPERTY PROPORTION'] = 100* x['PROPERTY PROPORTION'] / len(x['PROPERTY PROPORTION'])

# Plot the data and configure the settings
plt.plot(x['PROPERTY PROPORTION'],x['price'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to Price')
plt.xlabel('Percentage of Properties in ascending order of Price')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')
plt.show()


# Capture the necessary data
variables = data.columns

count = []

for variable in variables:
    length = data[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(data), 2)





# Correlation Matrix
# Compute the correlation matrix
d= data[['property_type','surface_total_in_m2','surface_covered_in_m2', 'price_usd_per_m2', 'price_per_m2', 'rooms', 'price']]
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=True, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of all Numerical Variables')
plt.show()


# Training data

columns = ['property_type','surface_total_in_m2','surface_covered_in_m2', 'price_usd_per_m2', 'price_per_m2', 'rooms', 'price']
data_model = data.loc[:,columns]

one_hot_features = ['property_type']

longest_str = max(one_hot_features, key=len)
total_num_unique_categorical = 0
for feature in one_hot_features:

     num_unique = len(data[feature].unique())
     print('{col:<{fill_col}} : {num:d} unique categorical values.'.format(col=feature,
				fill_col=len(longest_str),
				num=num_unique))
     total_num_unique_categorical += num_unique

one_hot_encoded = pd.get_dummies(data_model[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

#Delete the old columns...
data_model = data_model.drop(one_hot_features, axis=1)

plt.figure(figsize=(10,6))
sns.distplot(data_model['price'])
plt.title('Histogram of price')
plt.show()

# Take the log and normalise
data_model['price'] = StandardScaler().fit_transform(np.log(data_model['price']).reshape(-1,1))

plt.figure(figsize=(10,6))
sns.distplot(data_model['price'])
plt.title('Histogram of Normalised SALE PRICE')
plt.show()



# Remove observations with missing values
data['property_type'].notnull()
data_model = data_model['surface_covered_in_m2'].notnull().astype('int')

# Add 1 to Units (Some of the variables contain zeroes, which is why I need to add 1 so that I can take the log before normalising it - you can see that in the table below. Using the log allows me to get rid of the skew in the data and have a more normal distribution. The reason why I need to add 1 is because I can't take the log of 0 - it is not defined. The log of 1 however is.)

data_model['property_type'] = data_model['property_type'] + 1
data_model['surface_covered_in_m2'] = data_model['surface_covered_in_m2'] + 1
data_model['surface_total_in_m2'] = data_model['surface_total_in_m2'] + 1


data_model['property_type'] = StandardScaler().fit_transform(np.log(data_model['property_type']).reshape(-1,1))
data_model['surface_covered_in_m2'] = StandardScaler().fit_transform(np.log(data_model['surface_covered_in_m2']).reshape(-1,1))
data_model['surface_total_in_m2'] = StandardScaler().fit_transform(np.log(data_model['surface_total_in_m2']).reshape(-1,1))
