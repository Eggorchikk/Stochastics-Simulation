import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_excel(r"C:\Users\vcerc\OneDrive\Desktop\Uni\2DMM30 SS\Assignment\data_assignment.xlsx")
data['Date_Day'] = data['Date'].dt.date
data['Date_Time'] = data['Date'].dt.time
data['Week'] = data['Date'].dt.to_period('W').astype(str)
mask_income = data['Income/Expense'] == 'Income'
mask_expense = data['Income/Expense'] == 'Expense'
income_amount_df = data[mask_income]
expense_amount_df = data[mask_expense]
print(income_amount_df.describe())

grouped_category_in_df = income_amount_df.groupby('Category')


# sns.barplot(data = expense_amount_df, x = 'Category', y = 'Amount', ci=None)
# plt.show()


# heatmap_data = expense_amount_df.pivot_table(
#     index='Category',
#     columns='Week',
#     values='Amount',
#     aggfunc='sum',
#     fill_value=0
# )
# print(heatmap_data.head())
# sns.heatmap(data=heatmap_data, cmap='YlGnBu')

grouped_category_in_df.plot(kind="bar", stacked=True)

#sns.histplot(expense_amount_df['Amount'], bins=30)


plt.show()