### Statistics in Python https://scipy-lectures.org/packages/statistics/index.html#data-representation-and-interaction ###

# import needed packages #
import numpy as np
import scipy as sc
from scipy import stats

import matplotlib.pyplot as plt

import pandas as pd
import pandas.plotting as plotting

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

import seaborn as sns

import urllib
import urllib.request
import os

# Creating dataframes: reading data files or converting arrays #
# Reading from a CSV file # 
data = pd.read_csv('examples/brain_size.csv', sep=';', na_values=".")
data

# Manipulating data #
data.describe()
data.shape
data.columns
print(data['Gender'])
data[data['Gender'] == 'Female']['VIQ'].mean()

groupby_gender = data.groupby('Gender')
groupby_gender.first()
#loop?#
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
groupby_gender.mean()

# Exercise #
data.mean() # mean of VIQ for full population is 112.35
data['Gender'].value_counts() # female: 20; male: 20
groupby_gender['MRI_Count'].mean() # mean MRI for female: 862654.6; for male

# Plotting data #
groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])
# show all the figures above 
plt.show()

# Exercise???#
# plotting.scatter_matrix(groupby_gender['FSIQ', 'VIQ', "PIQ"])
# plt.show()

# Hypothesis testing: comparing two groups #
# 1-sample t-test: testing the value of a population mean #
stats.ttest_1samp(data['VIQ'], 0) # p-value is 1.33e-28, not equal to 0
# 2-sample t-test: testing for difference across populations #
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq) # p-value is 0.44 > 0.05, not sig, no difference 
# Paired tests: repeated measurements on the same individuals #
stats.ttest_ind(data['FSIQ'], data['PIQ']) # p-value is 0.64 > 0.05, not sig, no difference - wrong though due to same population/inter-subject variability
stats.ttest_rel(data['FSIQ'], data['PIQ']) # p-value is 0.08 > 0.05, not sig, no difference
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) # give the same result
stats.wilcoxon(data['FSIQ'], data['PIQ'])

# Exercise #
female_weight = data[data['Gender'] == 'Female']['Weight']
male_weight = data[data['Gender'] == 'Male']['Weight']
female_weight.isnull().sum()
male_weight.isnull().sum()
# remove nan in Weight var to avoid output error #
stats.ttest_ind(female_weight, male_weight) # this does not work due to missing values, need to drop NAN first
stats.stats.ttest_ind(female_weight.dropna(), male_weight.dropna()) # p-value is 2.23e-05 < 0.05, sig, there is a difference
# sc.stats.mannwhitneyu(female_weight, male_weight)

# Linear models, multiple factors, and analysis of variance #
# A simple linear regression #
x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
dt = pd.DataFrame({'x': x, 'y': y})
model = ols("y ~ x", dt).fit()
print(model.summary()) # coef is 2.94; se is 0.34; p < 0.01

# Categorical variables: comparing groups or multiple categories #
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary()) # coef is 5.80; se is 7.51; p = 0.45
# An integer column can be forced to be treated as categorical using
model = ols('VIQ ~ C(Gender)', data).fit()

# Link to t-tests between different FSIQ and PIQ #
data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)
model = ols("iq ~ type", data_long).fit()
print(model.summary())  
# get same results as using t-test
stats.ttest_ind(data['FSIQ'], data['PIQ'])

# Multiple Regression: including multiple factors #
data_iris = pd.read_csv('examples/iris.csv')
model = ols('sepal_width ~ name + petal_length', data_iris).fit()
print(model.summary())

# Post-hoc hypothesis testing: analysis of variance (ANOVA) #
print(model.f_test([0, 1, -1, 0])) # F = 3.25; p = 0.07, not sig


### More visualization: seaborn for statistical exploration ###
if not os.path.exists('wages.txt'):
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', 'wages.txt')

# Give names to the columns
names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]

short_names = [n.split(':')[0] for n in names]

data_1 = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None, header=None, engine='python')
data_1.columns = short_names

# Log-transform the wages, because they typically are increased with
# multiplicative factors
data_1['WAGE'] = np.log10(data_1['WAGE'])

sns.pairplot(data_1, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg')

sns.pairplot(data_1, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')
plt.suptitle('Effect of gender: 1=Female, 0=Male')
plt.show()

sns.pairplot(data_1, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='RACE')
plt.suptitle('Effect of race: 1=Other, 2=Hispanic, 3=White')
plt.show()

sns.pairplot(data_1, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='UNION')
plt.suptitle('Effect of union: 1=Union member, 0=Not union member')
plt.show()

# Plot a simple regression # 
sns.lmplot(y='WAGE', x='EDUCATION', data=data_1)
plt.show()
# Make regression less sensitive to outliers #
sns.lmplot(y='WAGE', x='EDUCATION', data=data_1, robust=True)
plt.show()

# Seaborn changes the default of matplotlib figures to achieve a more “modern”, “excel-like” look. It does that upon import. You can reset the default using: #
plt.rcdefaults()

### Testing for interactions ###
# result = sm.OLS(endog='wage', exog='education*gender', data=data_1).fit()
# print(result.summary())