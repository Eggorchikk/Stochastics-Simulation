import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns 

#extracting data
data = pd.read_excel("data_assignment.xlsx")
data = data.sort_values(['Category', 'Date'])
data['Interarrival_Time'] = data.groupby('Category')['Date'].diff().dt.total_seconds()
#added columns for date/time/week/month
data['Date_Day'] = data['Date'].dt.date
data['Date_Time'] = data['Date'].dt.time
data['Week'] = data['Date'].dt.to_period('W').astype(str)
data['Month'] = data['Date'].dt.to_period('M').astype(str)

#added masks for income/expense
mask_income = data['Income/Expense'] == 'Income'
mask_expense = data['Income/Expense'] == 'Expense'

#data for income/expense
income_amount_df = data[mask_income]
expense_amount_df = data[mask_expense]

income_amount = income_amount_df['Amount']
expense_amount = expense_amount_df['Amount']

#
#Creating pivot/heatmap df for income/expense
#
expense_pivot_data = expense_amount_df.pivot_table(
    index='Week',
    columns='Category',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)
#renaming index to week#No
expense_pivot_data.index = [f"W{i+1}" for i in range(len(expense_pivot_data.index))]

expense_heatmap_data = expense_amount_df.pivot_table(
    index='Category',
    columns='Week',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)
#renaming columns to week#No
expense_heatmap_data.columns = [f"W{i+1}" for i in range(len(expense_heatmap_data.columns))]

income_heatmap_data = income_amount_df.pivot_table(
    index='Category',
    columns='Week',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)
income_heatmap_data.columns = [f"W{i+1}" for i in range(len(income_heatmap_data.columns))]


income_pivot_data = income_amount_df.pivot_table(
    index='Week',
    columns='Category',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)
income_pivot_data.index = [f"W{i+1}" for i in range(len(income_pivot_data.index))]


#sort expenses/income by sum and ascending
expense_totals = expense_amount_df.groupby('Category')['Amount'].sum().sort_values(ascending=True)
income_totals = income_amount_df.groupby('Category')['Amount'].sum().sort_values(ascending=True)

print(income_amount_df.groupby('Category')['Amount'].describe())
print(expense_amount_df.groupby('Category')['Amount'].describe())

"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# INCOME/EXPENSE BARPLOT
income_totals.plot(kind='bar', ax=ax1, color='green')
ax1.set_title("Income by Category")
ax1.set_xlabel("Category")
ax1.set_ylabel("Amount")
ax1.tick_params(axis='x', rotation=45)

expense_totals.plot(kind='bar', ax=ax2, color='red')
ax2.set_title("Expenses by Category")
ax2.set_xlabel("Category")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""

# sns.boxplot(data=income_amount_df, x='Category', y='Amount')
# sns.boxplot(data=expense_amount_df, x='Category', y='Amount')

"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
#INCOME/EXPENSE HEATMAP
sns.heatmap(data=expense_heatmap_data, cmap='viridis', ax=ax1)
ax1.set_title("Weekly Expenses by Category")
ax1.tick_params(axis='x', rotation=45)

sns.heatmap(data=income_heatmap_data, cmap='viridis', ax=ax2)
ax2.set_title("Weekly income by Category")
ax2.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', rotation=0)

for ax in (ax1, ax2):
    ax.set_ylabel(None)
    ax.set_xlabel("Week")
fig.supylabel("Category")    

plt.tight_layout()
plt.show()"""


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
#INCOME/EXPENSE STACKBARPLOT
expense_pivot_data.plot(kind="bar", stacked=True, width=0.9, ax=ax1)

ax1.set_title("Weekly Expenses by Category")
ax1.set_xlabel("Week")
ax1.set_ylabel("Amount")
ax1.tick_params(axis='x', rotation=45)

income_pivot_data.plot(kind="bar", stacked=True, width=0.9, ax=ax2)
ax2.set_title("Weekly income by Category")
ax2.set_xlabel("Week")
ax2.set_ylabel("Amount")
ax2.tick_params(axis='x', rotation=45)
# move legend out of the way
ax1.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()






other_income_amount_df = income_amount_df[income_amount_df['Category'] == 'Other']['Amount']
food_expense_amount_df = expense_amount_df[expense_amount_df['Category'] == 'Food']['Amount']
#HISTOGRAM OF OTHER INCOME
plt.figure()
plt.hist(other_income_amount_df, bins=30, density=True)
plt.title('Income Distribution for Other Category')
plt.xlabel('Amount (€)')
plt.ylabel('Frequency')

#HISTOGRAM OF FOOD EXPENSE
plt.figure()
plt.hist(food_expense_amount_df, bins=30, density=True)
plt.title('Expense Distribution for Food Category')
plt.xlabel('Amount (€)')
plt.ylabel('Frequency')

food_expense_df = expense_amount_df[expense_amount_df['Category'] == 'Food']
inter_days_food = food_expense_df['Interarrival_Time'].dropna()/86400

transport_expense_df = expense_amount_df[expense_amount_df['Category'] == 'Transportation']
inter_days_transport = transport_expense_df['Interarrival_Time'].dropna()/86400

#HISTOGRAM INTERARRIVAL TIMES FOOD EXPENSE
plt.figure()
sns.histplot(inter_days_food, bins=20, stat='density', color='skyblue')
plt.xlabel('Interarrival time (seconds)')
plt.ylabel('Density')
plt.title('Interarrival Time Distribution – Food Expenses')

plt.figure()
sns.histplot(inter_days_transport, bins=20, stat='density', color='skyblue')
plt.xlabel('Interarrival time (seconds)')
plt.ylabel('Density')
plt.title('Interarrival Time Distribution – Food Expenses')
plt.show()

# # INCOME AMOUNT DISTRIBUTIONS
# plt.figure()
# sns.histplot(income_amount, bins=30)

# income_M1 = np.mean(income_amount)
# income_M2 = np.mean(income_amount**2)

# income_mu = income_M1
# income_sigma2 = income_M2 - income_M1**2

# # fit an exponential distribution
# lam = 1/income_M1
# fitExpDist = stats.expon(scale=1/lam)
# xs = np.arange(np.min(income_amount), np.max(income_amount), 0.1)
# ys = fitExpDist.pdf(xs)
# plt.figure()
# plt.hist(income_amount, bins=10, rwidth=0.8, density=True)
# plt.plot(xs, ys, color='red')

# # ecdf
# ecdfx = np.sort(income_amount)
# ecdfy = np.arange(1, len(income_amount)+1) / len(income_amount)
# plt.figure()
# plt.step(ecdfx, ecdfy, where='post', color='g')
# plt.plot(xs, fitExpDist.cdf(xs), color='orange')

# # Kolmogorov-Smirnov test
# tst1 = stats.kstest(income_amount, fitExpDist.cdf)
# print('KS Test Gamma distribution: ' + str(tst1))

# # fit a gamma distribution
# alpha = income_M1**2 / (income_M2 - income_M1**2)
# beta = income_M1 / (income_M2 - income_M1**2)
# fitGammaDist = stats.gamma(alpha, scale=1/beta)
# ys = fitGammaDist.pdf(xs)
# plt.plot(xs, fitGammaDist.cdf(xs), color='r')
# # Kolmogorov-Smirnov test
# tst1 = stats.kstest(income_amount, fitGammaDist.cdf)
# print('KS Test Exponential distribution: ' + str(tst1))

# # fit a negative binaomial distribution
# p = income_mu / income_sigma2
# r = income_mu**2 / (income_sigma2 - income_mu)
# fitNBDist = stats.nbinom(n=r, p=p)
# xs = np.arange(0, max(income_amount)+5)
# plt.plot(xs, fitNBDist.cdf(xs), 'b--', label='NB CDF')
# # Kolmogorov-Smirnov test (use CDF directly)
# ks_result = stats.kstest(income_amount, fitNBDist.cdf)
# print('KS Test Negative Binomial distribution:', ks_result)


# # EXPENSE AMOUNT DISTRIBUTIONS
# plt.figure()
# sns.histplot(expense_amount, bins=30)

# expense_M1 = np.mean(expense_amount)
# expense_M2 = np.mean(expense_amount**2)

# expense_mu = expense_M1
# expense_sigma2 = expense_M2 - expense_M1**2

# # fit an exponential distribution
# lam = 1/expense_M1
# fitExpDist = stats.expon(scale=1/lam)
# xs = np.arange(np.min(expense_amount), np.max(expense_amount), 0.1)
# ys = fitExpDist.pdf(xs)
# plt.figure()
# plt.hist(expense_amount, bins=30, rwidth=0.8, density=True)
# plt.plot(xs, ys, color='red')

# # ecdf
# ecdfx = np.sort(expense_amount)
# ecdfy = np.arange(1, len(expense_amount)+1) / len(expense_amount)
# plt.figure()
# plt.step(ecdfx, ecdfy, where='post', color='g')
# plt.plot(xs, fitExpDist.cdf(xs), color='orange')

# # Kolmogorov-Smirnov test
# tst1 = stats.kstest(expense_amount, fitExpDist.cdf)
# print('KS Test Gamma distribution: ' + str(tst1))

# # fit a gamma distribution
# alpha = expense_M1**2 / (expense_M2 - expense_M1**2)
# beta = expense_M1 / (expense_M2 - expense_M1**2)
# fitGammaDist = stats.gamma(alpha, scale=1/beta)
# ys = fitGammaDist.pdf(xs)
# plt.plot(xs, fitGammaDist.cdf(xs), color='r')
# # Kolmogorov-Smirnov test
# tst1 = stats.kstest(expense_amount, fitGammaDist.cdf)
# print('KS Test Exponential distribution: ' + str(tst1))

# # fit a negative binaomial distribution
# p = expense_mu / expense_sigma2
# r = expense_mu**2 / (expense_sigma2 - expense_mu)
# fitNBDist = stats.nbinom(n=r, p=p)
# xs = np.arange(0, max(expense_amount)+5)
# plt.plot(xs, fitNBDist.cdf(xs), 'b--', label='NB CDF')
# # Kolmogorov-Smirnov test (use CDF directly)
# ks_result = stats.kstest(expense_amount, fitNBDist.cdf)
# print('KS Test Negative Binomial distribution:', ks_result)


# def simulateCompoundPoissonProcess(lam, amount_dist, T):
#     arrival_times = []
#     t = 0
#     while t < T:
#         t += stats.expon(scale=1/lam).rvs()
#         if t < T:
#             arrival_times.append(t)
#     amounts = amount_dist.rvs(size=len(arrival_times))
#     cumulative = np.cumsum(amounts)
#     return np.array(arrival_times), amounts, cumulative

# T = 100 
# lam_income = 3
# lam_expense = 5

# # INCOME: Exponential distribution
# income_mean = 30
# income_dist = stats.expon(scale=income_mean)

# # EXPENSE OPTION 1: Exponential
# expense_mean = 20
# expense_dist_exp = stats.expon(scale=expense_mean)

# # EXPENSE OPTION 2: Negative Binomial with mean=20, var=50
# mu = 20
# sigma2 = 50
# p_nb = mu / sigma2
# r_nb = mu**2 / (sigma2 - mu)    
# expense_dist_nb = stats.nbinom(n=r_nb, p=p_nb)

# # === SIMULATE INCOME PROCESS ===
# arr_income, amt_income, cum_income = simulateCompoundPoissonProcess(lam_income, income_dist, T)

# # === SIMULATE EXPENSE PROCESS (choose one) ===
# # expense_dist = expense_dist_exp  # uncomment for exponential
# expense_dist = expense_dist_nb     # uncomment for negative binomial
# arr_expense, amt_expense, cum_expense = simulateCompoundPoissonProcess(lam_expense, expense_dist, T)

# # === NET BALANCE ===
# all_times = np.sort(np.unique(np.concatenate([arr_income, arr_expense, [T]])))
# income_at_times = np.interp(all_times, np.append(arr_income, T), np.append(cum_income, cum_income[-1] if len(cum_income) else 0))
# expense_at_times = np.interp(all_times, np.append(arr_expense, T), np.append(cum_expense, cum_expense[-1] if len(cum_expense) else 0))
# net_balance = income_at_times - expense_at_times

# # === PLOTS ===
# plt.figure(figsize=(10, 6))
# plt.step(arr_income, cum_income, label="Cumulative Income", where='post', color='green')
# plt.step(arr_expense, cum_expense, label="Cumulative Expenses", where='post', color='red')
# plt.step(all_times, net_balance, label="Net Balance", where='post', color='blue')
# plt.xlabel("Time")
# plt.ylabel("Amount (€)")
# plt.title("Compound Poisson Process – Incomes vs Expenses")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

# # --- Fixed Budget Goal Simulation ---
# def simulate_until_goal(income_lam, income_scale, expense_lam, expense_dist, goal):
#     t = 0
#     savings = 0
#     times = [0]
#     savings_list = [0]

#     while savings < goal:
#         next_income_time = np.random.exponential(1 / income_lam)
#         next_expense_time = np.random.exponential(1 / expense_lam)

#         if next_income_time < next_expense_time:
#             t += next_income_time
#             amount = np.random.exponential(income_scale)
#             savings += amount
#         else:
#             t += next_expense_time
#             amount = expense_dist.rvs()
#             savings -= amount

#         times.append(t)
#         savings_list.append(savings)

#     return times, savings_list

# # --- Seasonal Expenses Only Simulation ---
# def seasonal_expense_rate(day):
#     weekday = day % 7
#     return 1.5 if weekday in [5, 6] else 0.5  # Higher rate on weekends

# def simulate_seasonal_expenses(T_days, base_expense_dist):
#     t = 0
#     savings = 0
#     times = [0]
#     savings_list = [0]
    
#     while t < T_days:
#         day = int(t)
#         lam = seasonal_expense_rate(day)
#         t += np.random.exponential(1 / lam)
#         if t >= T_days:
#             break
#         expense = base_expense_dist.rvs()
#         savings -= expense
#         times.append(t)
#         savings_list.append(savings)

#     return times, savings_list

# # --- Run Simulations ---
# np.random.seed(42)

# # Parameters
# income_lambda = 0.8
# income_scale = 20
# expense_lambda = 1.2
# nbinom_r = 2
# nbinom_p = 0.4
# expense_dist = stats.nbinom(n=nbinom_r, p=nbinom_p)
# goal = 500

# # Simulate fixed goal
# times_goal, savings_goal = simulate_until_goal(income_lambda, income_scale, expense_lambda, expense_dist, goal)

# # Simulate seasonal expenses
# T_days = 60
# base_expense_dist = stats.expon(scale=15)
# times_seasonal, savings_seasonal = simulate_seasonal_expenses(T_days, base_expense_dist)

# # --- Plot Results ---
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# axs[0].plot(times_goal, savings_goal, label='Savings Over Time (Goal €500)')
# axs[0].axhline(500, color='red', linestyle='--', label='Goal €500')
# axs[0].set_title('Compound Poisson Process Until Goal Reached')
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('Savings (€)')
# axs[0].legend()

# axs[1].step(times_seasonal, savings_seasonal, where='post', color='orange')
# axs[1].set_title('Seasonal Compound Poisson Process (Expenses Only)')
# axs[1].set_xlabel('Time (days)')
# axs[1].set_ylabel('Cumulative Expenses (€)')

# plt.tight_layout()


# plt.show()