#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('marketing_data.csv')
print("Total rows and columns:", df.shape)


# In[3]:


print("\nFirst 5 rows of the dataset:")
print(df.head())


# In[4]:


print("\nMissing values in each column:")
print(df.isnull().sum())


# In[5]:


print("\nRemoving duplicate rows...")
print("Rows before removing duplicates:", len(df))

df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))


# In[6]:


df['date'] = pd.to_datetime(df['date'], errors='coerce')


# In[7]:


invalid_dates = df[df['date'].isnull()]
print("Rows with invalid dates (will be removed):")
print(invalid_dates)


# In[8]:


df = df[df['date'].isnull() == False]
print("Rows after removing bad dates:", len(df))


# In[9]:


print("\nHandling missing values in marketing_spend...")
print("Missing spend values:", df['marketing_spend'].isnull().sum())


# In[10]:


median_value = df['marketing_spend'].median()
print("Median spend value:", median_value)


# In[11]:


df['marketing_spend'] = df['marketing_spend'].fillna(median_value)
print("Missing values after filling:", df['marketing_spend'].isnull().sum())


# In[12]:


print("\nRemoving negative marketing spend rows...")
print("Rows with negative spend:")
print(df[df['marketing_spend'] < 0])


# In[13]:


df = df[df['marketing_spend'] >= 0]
print("Rows after removing negatives:", len(df))


# In[14]:


print("\nChecking business failure cases (spend > 0 but customers = 0):")
failure_cases = df[(df['marketing_spend'] > 0) & (df['new_customers'] == 0)]
print(failure_cases)


# In[15]:


print("\nDetecting outliers using IQR...")

Q1 = df['marketing_spend'].quantile(0.25)
Q3 = df['marketing_spend'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower bound =", lower_bound)
print("Upper bound =", upper_bound)


# In[16]:


outlier_rows = df[(df['marketing_spend'] > upper_bound) | (df['marketing_spend'] < lower_bound)]
print("\nOutlier rows found:")
print(outlier_rows)


# In[17]:


df['marketing_spend'] = df['marketing_spend'].clip(upper=upper_bound)
print("\nOutliers have been capped at:", upper_bound)
print("Final cleaned dataset shape:", df.shape)


# In[18]:


df['month'] = df['date'].dt.to_period('M')

monthly = df.groupby('month').agg(
    total_spend     = ('marketing_spend', 'sum'),
    total_customers = ('new_customers', 'sum')
).reset_index()


# In[19]:


monthly['month'] = monthly['month'].astype(str)
print(monthly)


# In[20]:


monthly['CAC'] = np.where(
    monthly['total_customers'] > 0,                       
    monthly['total_spend'] / monthly['total_customers'],   
    np.nan                                                
)

print("\nCAC calculated:")
print(monthly[['month', 'total_spend', 'total_customers', 'CAC']])


# In[23]:


monthly['CAC_verification'] = np.where(
    monthly['CAC'].notna(),
    monthly['CAC'] * monthly['total_customers'],
    np.nan
)

monthly['CAC_check_pass'] = np.where(
    monthly['CAC'].notna(),
    abs(monthly['CAC_verification'] - monthly['total_spend']) < 0.01,
    False
)

print("\nVerification check:")
print(monthly[['month', 'CAC', 'CAC_verification', 'CAC_check_pass']])


# In[22]:


def classify_cac(cac):
    if pd.isna(cac):
        return 'Corrupt Data'
    elif cac < 50:
        return 'Very Efficient'
    elif cac < 100:
        return 'Efficient'
    elif cac < 150:
        return 'Moderate'
    else:
        return 'Inefficient'

monthly['CAC_Classification'] = monthly['CAC'].apply(classify_cac)

spend_threshold = monthly['total_spend'].quantile(0.75)
monthly['High_Spend_Flag'] = monthly['total_spend'] > spend_threshold

print("High spend threshold:", round(spend_threshold, 2))
print(monthly[['month', 'total_spend', 'High_Spend_Flag', 'CAC', 'CAC_Classification']])


# In[23]:


def anomaly_reason(row):
    reasons = []
    if pd.isna(row['CAC']):
        reasons.append("Spend with zero customers — critical business failure")
    if row['CAC_Classification'] == 'Inefficient':
        reasons.append("CAC >= 150, inefficient spending")
    if row['High_Spend_Flag']:
        reasons.append("High spend month (top 25%)")
    return '; '.join(reasons) if reasons else 'Normal'

monthly['Anomaly_Reason'] = monthly.apply(anomaly_reason, axis=1)

print("\nBusiness Intelligence Table:")
print(monthly[['month', 'CAC', 'CAC_Classification', 'High_Spend_Flag', 'Anomaly_Reason']].to_string(index=False))


# In[24]:


monthly['CAC_Rolling_3M'] = monthly['CAC'].rolling(window=3).mean()

weights = np.arange(1, len(monthly) + 1)
monthly['CAC_Weighted'] = (monthly['CAC'].fillna(0) * weights) / weights.sum()

print("Trend Analysis:")
print(monthly[['month', 'CAC', 'CAC_Rolling_3M', 'CAC_Weighted']].to_string(index=False))


# In[25]:


first_cac = monthly['CAC'].dropna().iloc[0]
last_cac = monthly['CAC'].dropna().iloc[-1]

if last_cac > first_cac:
    print("\nTrend: CAC is INCREASING — acquisition is becoming less efficient over time.")
else:
    print("\nTrend: CAC is DECREASING — acquisition is becoming more efficient over time.")


# In[29]:


plt.figure(figsize=(12, 6))

plt.plot(monthly['month'].astype(str), monthly['CAC'],
         marker='o', label='Monthly CAC', color='steelblue', linewidth=2)

plt.plot(monthly['month'].astype(str), monthly['CAC_Rolling_3M'],
         label='3-Month Rolling Avg CAC', color='orange', linewidth=2, linestyle='--')

plt.plot(monthly['month'].astype(str), monthly['CAC_Weighted'],
         label='Weighted CAC', color='green', linewidth=2, linestyle='-.')

plt.title('Monthly Customer Acquisition Cost (CAC) Trend')
plt.xlabel('Month')
plt.ylabel('CAC')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('CAC_Trend.png', dpi=150)
plt.show()

print("Chart saved as CAC_Trend.png")


# In[30]:


monthly_total = monthly['total_spend'].sum()
raw_total     = df['marketing_spend'].sum()
difference    = abs(monthly_total - raw_total)

print("Total spend from monthly table:", monthly_total)
print("Total spend from cleaned data: ", raw_total)
print("Difference:                    ", difference)

if difference < 0.01:
    print("Integrity Check: PASS - Both totals match!")
else:
    print("Integrity Check: MISMATCH - Please check your data.")


# In[31]:


monthly.to_excel('marketing_report_22304071.xlsx', index=False)
monthly.to_csv('marketing_report_clean_22304071.csv', index=False)

print("Files exported successfully:")
print("  - marketing_report_22304071.xlsx")
print("  - marketing_report_clean_22304071.csv")

