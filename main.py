import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import scipy.stats as sc
import math as math

df = pd.read_excel('university data.xlsx', sheetname='university_data')

df1 = pd.read_excel(dataset, sheetname='university_data', usecols = [2,3,4,5])
df1 = df1[:-1]

CSScore = df.iloc[0:49,2]
ResearchOverhead = df.iloc[0:49,3]
AdminBasePay = df.iloc[0:49,4]
Tuition = df.iloc[0:49,5]

mu1 = np.mean(CSScore)
mu2 = np.mean(ResearchOverhead)
mu3 = np.mean(AdminBasePay)
mu4 = np.mean(Tuition)
muArr = [mu1, mu2, mu3, mu4]

var1 = np.var(CSScore)
var2 = np.var(ResearchOverhead)
var3 = np.var(AdminBasePay)
var4 = np.var(Tuition)

sigma1 = np.std(CSScore)
sigma2 = np.std(ResearchOverhead)
sigma3 = np.std(AdminBasePay)
sigma4 = np.std(Tuition)
sigmaArr = [sigma1, sigma2, sigma3, sigma4]

print('mu1 = ',round(mu1,3))
print('mu2 = ',round(mu2,3))
print('mu3 = ',round(mu3,3))
print('mu4 = ',round(mu4,3))

print('var1 = ',round(var1,3))
print('var2 = ',round(var2,3))
print('var3 = ',round(var3,3))
print('var4 = ',round(var4,3))

print('sigma1 = ',round(sigma1,3))
print('sigma2 = ',round(sigma2,3))
print('sigma3 = ',round(sigma3,3))
print('sigma4 = ',round(sigma4,3))

covarianceMat = df1.cov()
print('covarianceMat = ',round(covarianceMat,3))
df1.cov().hist()
mpl.show()

correlationMat = df1.corr()
print('correlationMat = ',round(correlationMat,3))
df1.corr().hist()
mpl.show()

logLikelihood = 0
logLikelihoodMulti = 0

df1 = pd.read_excel(dataset, sheetname='university_data', usecols = [2,3,4,5])
df1 = df1[:-1]
rowArr = []
for frame in range(len(df1)):
   rowArr = df1.values[frame]
   for i in range(len(rowArr)):
    logLikelihood+=math.log(sc.norm.pdf(rowArr[i], muArr[i], sigmaArr[i]))

print('logLikelihood = ',round(logLikelihood,3))

for frame in range(len(df1)):
   rowArr = df1.values[frame]
   logLikelihoodMulti += math.log(sc.multivariate_normal.pdf(rowArr, muArr, covarianceMat, allow_singular='True'))

print('logLikelihoodMulti = ',round(logLikelihoodMulti,3))
