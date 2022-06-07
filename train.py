import pandas as pd

from my_model.my_plot import plot_elbow_method, plot_silhouette_method

FIELD = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
gender_map = {0: 'Female', 1: 'Male'}

# Prepare Data
customer_df = pd.read_csv('Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy()

for i in range(len(customers)):
    if customers[i][0] == 'Female':
        customers[i][0] = 0
    elif customers[i][0] == 'Male':
        customers[i][0] = 1

print(FIELD)
print(customers)

plot_elbow_method(customers, 12)
plot_silhouette_method(customers, 12)
