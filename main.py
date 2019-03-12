import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('joined_transformed_v2.csv')
data = data.sample(n=100)
sigma = np.array((1, 0.6, 0.3,0.6, 1, 0.4, 0.3, 0.4, 1)).reshape((3, 3))
totalLoss = np.zeros(100000)

def computeTotalLoss(df, sigma):
    z = np.random.multivariate_normal([0, 0, 0], sigma)
    #add new columns to dataframe
    df['eps'] = np.random.normal(size = len(df['Loan']))
    df['Xi'] = df['Alpha']*z[df['Region']] + df['Gamma']*df['eps']
    
    #boolean mask
    df['Default'] = df['Xi'] < df['Threshold']
    
    #multiplies by 0 if X not under threshold. Otherwise multiplies by 1. 
    df['Loss'] = df['LGD']*df['EAD']*df['Default']
    return np.sum(df['Loss'])

def run_simulation():
    for i in range(100000):
        if(i%1000 == 0): print(i)
        df = data.copy()
        totalLoss[i] = computeTotalLoss(df, sigma)
    pd.Series(totalLoss).to_csv('loss.csv', index=False)

run_simulation()

