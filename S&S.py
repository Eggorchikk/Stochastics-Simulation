import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_excel("data_assignment.xlsx")
data['Date_Day'] = data['Date'].dt.date
data['Date_Time'] = data['Date'].dt.time
data['Week'] = data['Date'].dt.to_period('W').astype(str)
data['Month'] = data['Date'].dt.to_period('M').astype(str)
mask_income = data['Income/Expense'] == 'Income'
mask_expense = data['Income/Expense'] == 'Expense'
income_amount_df = data[mask_income]
expense_amount_df = data[mask_expense]
income_amount = income_amount_df['Amount']
expense_amount = expense_amount_df['Amount']
print(income_amount_df.describe())

grouped_category_in_df = income_amount_df.groupby('Category')

pivot_data = expense_amount_df.pivot_table(
    index='Week',
    columns='Category',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)

heatmap_data = expense_amount_df.pivot_table(
    index='Category',
    columns='Week',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)

# plt.figure()
# sns.barplot(data = expense_amount_df, x = 'Category', y = 'Amount', ci=None)

# print(heatmap_data.head())
# plt.figure()
# sns.heatmap(data=heatmap_data, cmap='YlGnBu')

# plt.figure()
# pivot_data.plot(kind="bar", stacked=True)

# plt.figure()
# sns.histplot(expense_amount, bins=30)

income_M1 = np.mean(income_amount)
income_M2 = np.mean(income_amount**2)

# normal distribution
mu = income_M1
sigma2 = income_M2 - income_M1**2
fitNormDist = stats.norm(mu, np.sqrt(sigma2))

# Ass theoretical density???
xs = np.arange(np.min(income_amount), np.max(income_amount), 0.1)
ys = fitNormDist.pdf(xs)

plt.figure()
plt.hist(income_amount, bins=10, rwidth=0.8, density=True)
plt.plot(xs, ys, color='red')

# ecdf
ecdfx = np.sort(income_amount)
ecdfy = np.arange(1, len(income_amount)+1) / len(income_amount)
plt.figure()
plt.step(ecdfx, ecdfy, where='post')
plt.plot(xs, fitNormDist.cdf(xs), color='black')



plt.show()