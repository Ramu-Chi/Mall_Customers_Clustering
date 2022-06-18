'''
Detect possible outliers in the dataset:

Using the IQR method on each attributes that is chosen to cluster
'''
import numpy as np
import pandas as pd

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Interquartile Range (IQR) method for finding outliers
Q1 = []
Q3 = []
for i in range(len(FIELD)):
    min_i = min(customers[:, i])
    max_i = max(customers[:, i])

    Q1.append( np.percentile(customers[:, i], 25) )
    Q3.append( np.percentile(customers[:, i], 75) )
    IQR = Q3[i] - Q1[i]
    iqr_method_range = [Q1[i] - 1.5 * IQR, Q3[i] + 1.5 * IQR]

    print('%s' % FIELD[i])
    print('\tattributes range: ', [min_i, max_i])
    print('\tIQR method range: ', iqr_method_range, '\n')

print('-' * 10)

# Only Annual Income has value outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
IQR = Q3[1] - Q1[1] # IQR of Annual Income
print('Possible Outliers:')
for i, cus in enumerate(customers):
    if cus[1] < Q1[1] - 1.5 * IQR or cus[1] > Q3[1] + 1.5 * IQR:
        print('customer id %d:' % (i + 1), cus)
