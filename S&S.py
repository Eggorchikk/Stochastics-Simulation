import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns 

#extracting data
data = pd.read_excel("data_assignment.xlsx")
# data cleaning
data = data.dropna()
data = data[data["Amount"] >= 0]
data = data.drop_duplicates()
# data sorting
data = data.sort_values(['Category', 'Date'])
# data grouping
data['Interarrival_Time'] = data.groupby('Category')['Date'].diff().dt.total_seconds() / 86400
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


"""# Box plots
# grouping categories with small number of measurements
category_counts = expense_amount_df['Category'].value_counts()
small_categories = category_counts[category_counts < 5].index
expense_amount_df['Category_grouped'] = expense_amount_df['Category'].apply(lambda x: 'Other' if x in small_categories else x)

plt.figure(figsize=(10, 6))
sns.boxplot(y='Category_grouped', x='Amount', data=expense_amount_df)
plt.xscale('log')
plt.xlabel('Amount (log scale)')
plt.title('Expenses by Category (Log Scale) with Small Groups Combined')
plt.tight_layout()

#box plot of 'other' income
plt.figure()
sns.boxplot(x=income_amount_df[income_amount_df['Category'] == 'Other']['Amount']
)
plt.title('Income Distribution of "Other" category')
plt.xticks(rotation=45)
plt.tight_layout()"""

#remove income 'other' outliers
income_amount_df = income_amount_df[~((income_amount_df['Category'] == 'Other') & (income_amount_df['Amount'] > 100))]

#print(income_amount_df.groupby('Category')['Amount'].describe())
#print(expense_amount_df.groupby('Category')['Amount'].describe())



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

income_pivot_data = income_amount_df.pivot_table(
    index='Week',
    columns='Category',
    values='Amount',
    aggfunc='sum',
    fill_value=0
)
income_pivot_data.index = [f"W{i+1}" for i in range(len(income_pivot_data.index))]

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


#sort expenses/income by sum and ascending
expense_totals = expense_amount_df.groupby('Category')['Amount'].sum().sort_values(ascending=True)
income_totals = income_amount_df.groupby('Category')['Amount'].sum().sort_values(ascending=True)



# Weekly total income
weekly_income = income_amount_df.groupby('Week')['Amount'].sum().reset_index()
weekly_expense = expense_amount_df.groupby('Week')['Amount'].sum().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7), sharey=True)
sns.boxplot(y='Amount', data=weekly_income, ax=ax1, color='skyblue')
plt.suptitle('Weekly Total Income/Expense')
ax1.set_ylabel('Amount')
ax1.set_xlabel("Income")
sns.boxplot(y='Amount', data=weekly_expense, ax=ax2, color='skyblue')
ax2.set_xlabel("Expense")
plt.xticks(rotation=45)
plt.tight_layout()


# INCOME/EXPENSE BARPLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
income_totals.plot(kind='bar', ax=ax1, color='royalblue')
ax1.set_title("Income by Category")
ax1.set_xlabel("Category")
ax1.set_ylabel("Amount")
ax1.tick_params(axis='x', rotation=45)

expense_totals.plot(kind='bar', ax=ax2, color='orange')
ax2.set_title("Expenses by Category")
ax2.set_xlabel("Category")
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()

# Weekly analysis
expense_amount_df = expense_amount_df.copy()
# Add weekday
expense_amount_df['Weekday'] = expense_amount_df['Date'].dt.day_name()
# Total expense per day
daily_expense = expense_amount_df.groupby(['Date_Day', 'Weekday'])['Amount'].sum().reset_index()
# Average daily expense per weekday
avg_daily_expense = daily_expense.groupby('Weekday')['Amount'].mean()
# Define weekday order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_daily_expense = avg_daily_expense.reindex(weekday_order)

# Plot bar chart
plt.figure(figsize=(8,5))
sns.barplot(x=avg_daily_expense.index, y=avg_daily_expense.values, palette='viridis')

plt.title('Average Daily Expense per Weekday')
plt.ylabel('Average Expense')
plt.xlabel('Weekday')
plt.xticks(rotation=45)
plt.tight_layout()


# Total expense per day, per weekday, per category
daily_expense_cat = expense_amount_df.groupby(['Date_Day', 'Weekday', 'Category'])['Amount'].sum().reset_index()
# Average daily expense per weekday and category
avg_daily_expense_cat = daily_expense_cat.groupby(['Weekday', 'Category'])['Amount'].mean().reset_index()
# Pivot for stacked bar plot
avg_daily_expense_pivot = avg_daily_expense_cat.pivot(index='Weekday', columns='Category', values='Amount')
# Reorder by weekdays
avg_daily_expense_pivot = avg_daily_expense_pivot.reindex(weekday_order)

# Plot
avg_daily_expense_pivot.plot(kind='bar', stacked=True, figsize=(10,6), colormap='tab20')
plt.title('Average Daily Expense per Weekday by Category (Stacked)')
plt.ylabel('Average Expense')
plt.xlabel('Weekday')
plt.xticks(rotation=45)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


#INCOME/EXPENSE HEATMAP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
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


#INCOME/EXPENSE STACKBARPLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
expense_pivot_data.plot(kind="bar", stacked=True, width=0.9, ax=ax1, colormap='tab10')

ax1.set_title("Weekly Expenses by Category")
ax1.set_xlabel("Week")
ax1.set_ylabel("Amount")
ax1.tick_params(axis='x', rotation=45)

income_pivot_data.plot(kind="bar", stacked=True, width=0.9, ax=ax2, colormap='tab10')
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
household_expense_amount_df = expense_amount_df[expense_amount_df['Category'] == 'Household']['Amount']
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.histplot(inter_days_food, bins=20, stat='probability', color='skyblue', ax=ax1)
sns.histplot(inter_days_transport, bins=20, stat='density', color='skyblue', ax=ax2)
ax1.set_title("Interarrival Time Distribution – Food Expenses")
ax1.set_xlabel("Interarrival time (days)")
ax1.set_ylabel("Density")

ax2.set_title("Interarrival Time Distribution – Transport Expenses")
ax2.set_xlabel("Interarrival time (days)")
ax2.set_ylabel("Density")
plt.tight_layout()
plt.show()


# INCOME/EXPENSE INTERARRIVAL DISTRIBUTIONS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(income_amount['Interarrival_Time'], bins=30, ax=ax1)
ax1.set_title("Income Amount Distribution")
ax1.set_xlabel("Amount")
ax1.set_ylabel("Frequency")

sns.histplot(expense_amount['Interarrival_Time'], bins=30, ax=ax2)
ax2.set_title("Expense Amount Distribution")
ax2.set_xlabel("Amount")
ax2.set_ylabel("Frequency")
plt.show()



income_M1 = np.mean(income_amount_df['Interarrival_Time'])
income_M2 = np.mean(income_amount_df['Interarrival_Time']**2)

income_mu = income_M1
income_sigma2 = income_M2 - income_M1**2

# fit an exponential distribution TO INTERARRIVAL
lam = 1/income_M1
fitExpDist = stats.expon(scale=1/lam)
xs = np.arange(np.min(income_amount_df['Interarrival_Time']), np.max(income_amount_df['Interarrival_Time']), 0.1)
ys = fitExpDist.pdf(xs)
plt.figure()
plt.hist(income_amount_df['Interarrival_Time'], bins=10, rwidth=0.8, density=True, alpha=0.6, label='Income Histogram')  # Added alpha and label
plt.plot(xs, ys, color='red', lw=2, label='Exponential PDF')  # Added line width and label
plt.title('Income Interarrival with Exponential Fit')  # Added title
plt.xlabel('Income Interarrival')  # Added x-axis label
plt.ylabel('Density')        # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# ecdf
ecdfx = np.sort(income_amount_df['Interarrival_Time'])
ecdfy = np.arange(1, len(income_amount_df['Interarrival_Time'])+1) / len(income_amount_df['Interarrival_Time'])
plt.figure()
plt.step(ecdfx, ecdfy, where='post', color='g', label='Empirical CDF')  # Added label
plt.plot(xs, fitExpDist.cdf(xs), color='orange', lw=2, label='Exponential CDF')  # Added line width and label
plt.title('Income Interarrival ECDF and Exponential CDF')  # Added title
plt.xlabel('Income Interarrival Times')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# fit a gamma distribution
alpha = income_M1**2 / (income_M2 - income_M1**2)
beta = income_M1 / (income_M2 - income_M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
ys = fitGammaDist.pdf(xs)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', lw=2, label='Gamma CDF')  # Added line width and label
plt.title('Income Interarrival ECDF and Gamma CDF')  # Added title (to existing figure)
plt.xlabel('Income Interarrival Times')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# fit a negative binaomial distribution
p = income_mu / income_sigma2
r = income_mu**2 / (income_sigma2 - income_mu)
fitNBDist = stats.nbinom(n=r, p=p)
xs = np.arange(0, income_amount_df['Interarrival_Time'].dropna().max()+5)
plt.plot(xs, fitNBDist.cdf(xs), 'b--', label='NB CDF', lw=2)  # Added label and line width
plt.title('Income Interarrival ECDF and Negative Binomial CDF')  # Added title
plt.xlabel('Interarrival Times')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# Kolmogorov-Smirnov tests
clean_data = income_amount_df['Interarrival_Time'].dropna()
tst1 = stats.kstest(clean_data, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
tst2 = stats.kstest(clean_data, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst2))
tst3 = stats.kstest(clean_data, fitNBDist.cdf)
print('KS Test Negative Binomial distribution:', tst3)


clean_expense = expense_amount_df['Interarrival_Time'].dropna()

# --- Moments ---
M1 = np.mean(clean_expense)
M2 = np.mean(clean_expense**2)
mu = M1
sigma2 = M2 - M1**2

# --- Fit Exponential Distribution ---
lam = 1 / M1
fitExpDist = stats.expon(scale=1/lam)

# --- Fit Gamma Distribution ---
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)

# --- Fit Negative Binomial Distribution ---
p = mu / sigma2
r = mu**2 / (sigma2 - mu)
fitNBDist = stats.nbinom(n=r, p=p)

# --- Create x-values for plotting ---
xs = np.arange(0, clean_expense.max() + 5, 0.1)

# --- Empirical CDF ---
ecdfx = np.sort(clean_expense)
ecdfy = np.arange(1, len(clean_expense) + 1) / len(clean_expense)

# --- Plot ECDF with all fitted CDFs ---
plt.figure(figsize=(8, 5))
plt.step(ecdfx, ecdfy, where='post', color='g', label='Empirical CDF')
plt.plot(xs, fitExpDist.cdf(xs), color='orange', lw=2, label='Exponential CDF')
plt.plot(xs, fitGammaDist.cdf(xs), color='r', lw=2, label='Gamma CDF')
plt.plot(xs, fitNBDist.cdf(xs), 'b--', lw=2, label='Negative Binomial CDF')
plt.title('Income Interarrival ECDF with Theoretical CDFs')
plt.xlabel('Interarrival Time (days)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

tst1 = stats.kstest(clean_expense, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
tst2 = stats.kstest(clean_expense, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst2))
tst3 = stats.kstest(clean_expense, fitNBDist.cdf)
print('KS Test Negative Binomial distribution:', str(tst3))

#income fitting
income_M1 = np.mean(income_amount)
income_M2 = np.mean(income_amount**2)

income_mu = income_M1
income_sigma2 = income_M2 - income_M1**2

# fit an exponential distribution
lam = 1/income_M1
fitExpDist = stats.expon(scale=1/lam)
xs = np.arange(np.min(income_amount['Interarrival_Time']), np.max(income_amount), 0.1)
ys = fitExpDist.pdf(xs)
plt.figure()
plt.hist(income_amount, bins=10, rwidth=0.8, density=True, alpha=0.6, label='Income Histogram')  # Added alpha and label
plt.plot(xs, ys, color='red', lw=2, label='Exponential PDF')  # Added line width and label
plt.title('Income Amount with Exponential Fit')  # Added title
plt.xlabel('Income Amount')  # Added x-axis label
plt.ylabel('Density')        # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# ecdf
ecdfx = np.sort(income_amount)
ecdfy = np.arange(1, len(income_amount)+1) / len(income_amount)
plt.figure()
plt.step(ecdfx, ecdfy, where='post', color='g', label='Empirical CDF')  # Added label
plt.plot(xs, fitExpDist.cdf(xs), color='orange', lw=2, label='Exponential CDF')  # Added line width and label
plt.title('Income Amount ECDF and Exponential CDF')  # Added title
plt.xlabel('Income Amount')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# fit a gamma distribution
alpha = income_M1**2 / (income_M2 - income_M1**2)
beta = income_M1 / (income_M2 - income_M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
ys = fitGammaDist.pdf(xs)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', lw=2, label='Gamma CDF')  # Added line width and label
plt.title('Income Amount ECDF and Gamma CDF')  # Added title (to existing figure)
plt.xlabel('Income Amount')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# fit a negative binaomial distribution
p = income_mu / income_sigma2
r = income_mu**2 / (income_sigma2 - income_mu)
fitNBDist = stats.nbinom(n=r, p=p)
xs = np.arange(0, max(income_amount)+5)
plt.plot(xs, fitNBDist.cdf(xs), 'b--', label='NB CDF', lw=2)  # Added label and line width
plt.title('Income Amount ECDF and Negative Binomial CDF')  # Added title
plt.xlabel('Income Amount')  # Added x-axis label
plt.ylabel('CDF')            # Added y-axis label
plt.legend()                 # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# Kolmogorov-Smirnov tests
tst1 = stats.kstest(income_amount, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
tst2 = stats.kstest(income_amount, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst2))
tst3 = stats.kstest(income_amount, fitNBDist.cdf)
print('KS Test Negative Binomial distribution:', tst3)


#Expense fitting

expense_M1 = np.mean(expense_amount)
expense_M2 = np.mean(expense_amount**2)

expense_mu = expense_M1
expense_sigma2 = expense_M2 - expense_M1**2

# fit an exponential distribution
lam = 1/expense_M1
fitExpDist = stats.expon(scale=1/lam)
xs = np.arange(np.min(expense_amount), np.max(expense_amount), 0.1)
ys = fitExpDist.pdf(xs)
plt.figure()
plt.hist(expense_amount, bins=30, rwidth=0.8, density=True, alpha=0.6, label='Expense Histogram')  # Added alpha and label
plt.plot(xs, ys, color='red', lw=2, label='Exponential PDF')  # Added line width and label
plt.title('Expense Amount with Exponential Fit')  # Added title
plt.xlabel('Expense Amount')  # Added x-axis label
plt.ylabel('Density')         # Added y-axis label
plt.legend()                  # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid

# ecdf
ecdfx = np.sort(expense_amount)
ecdfy = np.arange(1, len(expense_amount)+1) / len(expense_amount)
plt.figure()
plt.step(ecdfx, ecdfy, where='post', color='g', label='Empirical CDF')  # Added label
plt.plot(xs, fitExpDist.cdf(xs), color='orange', lw=2, label='Exponential CDF')  # Added line width and label
plt.title('Expense Amount ECDF and Exponential CDF')  # Added title
plt.xlabel('Expense Amount')  # Added x-axis label
plt.ylabel('CDF')              # Added y-axis label
plt.legend()                   # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid


# fit a gamma distribution
alpha = expense_M1**2 / (expense_M2 - expense_M1**2)
beta = expense_M1 / (expense_M2 - expense_M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
ys = fitGammaDist.pdf(xs)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', lw=2, label='Gamma CDF')  # Added line width and label
plt.title('Expense Amount ECDF and Gamma CDF')  # Added title (to existing figure)
plt.xlabel('Expense Amount')  # Added x-axis label
plt.ylabel('CDF')              # Added y-axis label
plt.legend()                   # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid


# fit a negative binaomial distribution
p = expense_mu / expense_sigma2
r = expense_mu**2 / (expense_sigma2 - expense_mu)
fitNBDist = stats.nbinom(n=r, p=p)
xs = np.arange(0, max(expense_amount)+5)
plt.plot(xs, fitNBDist.cdf(xs), 'b--', label='NB CDF', lw=2)  # Added label and line width
plt.title('Expense Amount ECDF and Negative Binomial CDF')  # Added title
plt.xlabel('Expense Amount')  # Added x-axis label
plt.ylabel('CDF')              # Added y-axis label
plt.legend()                  # Added legend
plt.grid(True, linestyle='--', alpha=0.6)  # Added grid


# Kolmogorov-Smirnov test
tst1 = stats.kstest(expense_amount, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# Kolmogorov-Smirnov test
tst2 = stats.kstest(expense_amount, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst2))
# Kolmogorov-Smirnov test (use CDF directly)
tst3 = stats.kstest(expense_amount, fitNBDist.cdf)
print('KS Test Negative Binomial distribution:', tst3)




# Simulation using compound Poisson process
def simulateCompoundPoissonProcess(lam, amount_dist, T):
    arrival_times = []
    t = 0
    while t < T:
        t += stats.expon(scale=1/lam).rvs()
        if t < T:
            arrival_times.append(t)
    amounts = amount_dist.rvs(size=len(arrival_times))
    cumulative = np.cumsum(amounts)
    return np.array(arrival_times), amounts, cumulative

T = 100 

income_data = data[data['Income/Expense'] == 'Income']
income_data = income_data.sort_values(by='Date')
income_data['Interarrival_Time'] = income_data['Date'].diff().dt.total_seconds() / 86400
interarrival_times_income = income_data['Interarrival_Time'].dropna()
# Compute mean interarrival time
mean_interarrival_income = interarrival_times_income.mean()
# Compute lambda (arrival rate)
lam_income = 1 / mean_interarrival_income

expense_data = data[data['Income/Expense'] == 'Expense']
expense_data = expense_data.sort_values(by='Date')
expense_data['Interarrival_Time'] = expense_data['Date'].diff().dt.total_seconds() / 86400
interarrival_times_expense = expense_data['Interarrival_Time'].dropna()
# Compute mean interarrival time
mean_interarrival_expense = interarrival_times_expense.mean()
# Compute lambda (arrival rate)
lam_expense = 1 / mean_interarrival_expense


# INCOME: Gamma distribution
income_mean = income_amount.mean()
income_var = income_amount.var()
alpha_income = income_mean**2 / income_var
beta_income = income_mean / income_var
income_dist = stats.gamma(a=alpha_income, scale=1/beta_income)

# Expense: Gamma distribution
expense_mean = expense_amount.mean()
expense_var = expense_amount.var()
alpha_expense = expense_mean**2 / expense_var
beta_expense = expense_mean / expense_var
expense_dist = stats.gamma(a=alpha_expense, scale=1/beta_expense)


# SIMULATE INCOME/EXPENSE PROCESSES
arr_income, amt_income, cum_income = simulateCompoundPoissonProcess(lam_income, income_dist, T)
arr_expense, amt_expense, cum_expense = simulateCompoundPoissonProcess(lam_expense, expense_dist, T)

# NET BALANCE
all_times = np.sort(np.unique(np.concatenate([arr_income, arr_expense, [T]])))
income_at_times = np.interp(all_times, np.append(arr_income, T), np.append(cum_income, cum_income[-1] if len(cum_income) else 0))
expense_at_times = np.interp(all_times, np.append(arr_expense, T), np.append(cum_expense, cum_expense[-1] if len(cum_expense) else 0))
net_balance = income_at_times - expense_at_times

# PLOTS
plt.figure(figsize=(10, 6))
plt.step(arr_income, cum_income, label="Cumulative Income", where='post', color='green')
plt.step(arr_expense, cum_expense, label="Cumulative Expenses", where='post', color='red')
plt.step(all_times, net_balance, label="Net Balance", where='post', color='blue')
plt.xlabel("Time")
plt.ylabel("Amount (€)")
plt.title("Compound Poisson Process – Incomes vs Expenses")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Scenario functions

def increase_transport_expenses(df, increase_pct=0.2):
    df_scenario = df.copy()
    mask = (df_scenario['Category'] == 'Transportation') & (df_scenario['Income/Expense'] == 'Expense')
    df_scenario.loc[mask, 'Amount'] *= (1 + increase_pct)
    return df_scenario

def stop_other_income_for_one_month(df, month_str='2023-05'):
    df_scenario = df.copy()
    mask = (df_scenario['Category'] == 'Other') & (df_scenario['Income/Expense'] == 'Income') & (df_scenario['Date'].dt.to_period('M').astype(str) == month_str)
    df_scenario.loc[mask, 'Amount'] = 0
    return df_scenario

def add_monthly_allowance(df, allowance=100):
    df_scenario = df.copy()
    months = df_scenario['Date'].dt.to_period('M').unique()
    allowance_entries = []
    for month in months:
        # Add allowance on the 1st day of each month
        allowance_entries.append({
            'Date': month.to_timestamp(),
            'Income/Expense': 'Income',
            'Category': 'Allowance',
            'Amount': allowance
        })
    allowance_df = pd.DataFrame(allowance_entries)
    allowance_df['Date'] = pd.to_datetime(allowance_df['Date'])
    # Append allowance rows
    df_scenario = pd.concat([df_scenario, allowance_df], ignore_index=True)
    return df_scenario


# --- Prepare baseline data ---

data['Date'] = pd.to_datetime(data['Date'])

# Apply scenarios by creating scenario-specific dataframes
data_expense_transport_inc = increase_transport_expenses(data)
data_income_stop_other = stop_other_income_for_one_month(data)
data_income_with_allowance = add_monthly_allowance(data)

# Function to prepare inputs and simulate from a given dataframe
def simulate_from_data(df, T):

    # Simulate compound Poisson processes
    arr_income, amt_income, cum_income = simulateCompoundPoissonProcess(lam_income, income_dist, T)
    arr_expense, amt_expense, cum_expense = simulateCompoundPoissonProcess(lam_expense, expense_dist, T)

    # Calculate net balance over time
    all_times = np.sort(np.unique(np.concatenate([arr_income, arr_expense, [T]])))
    income_at_times = np.interp(all_times, np.append(arr_income, T), np.append(cum_income, cum_income[-1] if len(cum_income) else 0))
    expense_at_times = np.interp(all_times, np.append(arr_expense, T), np.append(cum_expense, cum_expense[-1] if len(cum_expense) else 0))
    net_balance = income_at_times - expense_at_times

    return all_times, cum_income, cum_expense, net_balance, arr_income, arr_expense

# Run baseline simulation
all_times_base, cum_income_base, cum_expense_base, net_balance_base, arr_income_base, arr_expense_base = simulate_from_data(data, T)
# Run scenario 1: Increase transportation expenses by 20%
all_times_s1, cum_income_s1, cum_expense_s1, net_balance_s1, arr_income_s1, arr_expense_s1 = simulate_from_data(data_expense_transport_inc, T)
# Run scenario 2: Stop "Other" income for one month (May 2023)
all_times_s2, cum_income_s2, cum_expense_s2, net_balance_s2, arr_income_s2, arr_expense_s2 = simulate_from_data(data_income_stop_other, T)
# Run scenario 3: Add extra monthly allowance (€100)
all_times_s3, cum_income_s3, cum_expense_s3, net_balance_s3, arr_income_s3, arr_expense_s3 = simulate_from_data(data_income_with_allowance, T)

# Plot baseline and scenarios
plt.figure(figsize=(12, 8))
# Baseline
plt.step(all_times_base, net_balance_base, label='Baseline Net Balance', where='post', color='blue')
# Scenario 1
plt.step(all_times_s1, net_balance_s1, label='Scenario 1: +20% Transport Expenses', where='post', color='red')
# Scenario 2
plt.step(all_times_s2, net_balance_s2, label='Scenario 2: Stop Other Income (May 2023)', where='post', color='orange')
# Scenario 3
plt.step(all_times_s3, net_balance_s3, label='Scenario 3: +€100 Monthly Allowance', where='post', color='green')

plt.xlabel('Time')
plt.ylabel('Net Balance (€)')
plt.title('Net Balance Over Time: Baseline vs Scenarios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Fixed Budget Goal Simulation
def simulate_until_goal(lam_income, income_dist, lam_expense, expense_dist, goal):
    t = 0
    savings = 0
    times = [0]
    savings_list = [0]

    while savings < goal:
        next_income_time = np.random.exponential(1 / lam_income)
        next_expense_time = np.random.exponential(1 / lam_expense)

        if next_income_time < next_expense_time:
            t += next_income_time
            amount = income_dist.rvs()
            savings += amount
        else:
            t += next_expense_time
            amount = expense_dist.rvs()
            savings -= amount

        times.append(t)
        savings_list.append(savings)

    return times, savings_list

# Parameters
# np.random.seed(42)
goal = 500

# Simulate
times_goal, savings_goal = simulate_until_goal(lam_income, income_dist, lam_expense, expense_dist, goal)

# Plot Results
plt.figure(figsize=(10, 6))

plt.plot(times_goal, savings_goal, label='Savings Over Time (Goal €500)')
plt.axhline(500, color='red', linestyle='--', label='Goal €500')
plt.title('Compound Poisson Process Until Goal Reached')
plt.xlabel('Time')
plt.ylabel('Savings (€)')
plt.legend()
plt.grid(True)


plt.show()