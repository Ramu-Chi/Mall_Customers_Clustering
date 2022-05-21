import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIELD = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
gender_map = {0: 'Female', 1: 'Male'}

# Prepare Data
customer_df = pd.read_csv('Mall_Customers.csv')
customer = customer_df[FIELD].to_numpy()

for i in range(len(customer)):
    if customer[i][0] == 'Female':
        customer[i][0] = 0
    elif customer[i][0] == 'Male':
        customer[i][0] = 1

print(FIELD)
print(customer)
