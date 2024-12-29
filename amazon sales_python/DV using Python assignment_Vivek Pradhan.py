#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### EDA

# In[2]:


df = pd.read_csv(r"D:\Data_analytics\Tableau\data\Amazon Sales data.csv")


# In[3]:


df.head()


# In[4]:


df.corr(numeric_only = True)


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.sample()


# In[8]:


df.shape


# In[9]:


df.isna().sum()


# ## Area chart

# In[10]:


df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors = 'coerce')
df['year'] = df['Ship Date'].dt.year

df.head()


# In[11]:


df_area = df.groupby([df['year'], 'Sales Channel'])['Total Cost'].sum().unstack()

df_area = df_area.sort_index()
df_area


# ### Insights
# 
# This plot shows the distribution of revenue (or another metric) over time, split by sales channel.  
# 
# 
# - The orange region (Offline) indicates periods when offline sales contributed more revenue than online sales.
# - Peaks in the data show spikes in revenue (or activity), which could be linked to events like seasonal sales or promotions.
# - Observe if Online (blue) revenue is steadily growing, which could indicate a shift in consumer behavior towards e-commerce.
# - If Online is growing, focus more resources on strengthening the digital presence.
# - Investigate causes behind peaks or declines in both sales channels for strategic improvements.

# In[12]:


plt.figure(figsize = (10, 6))
df_area.plot.area(figsize = (12, 6))



# ## Line chart
# 
# #### Insights
# 
# - The line chart gives a clearer view of trends over time for Online vs. Offline sales.
# 
# - Crossovers where Online sales surpass Offline sales.
# - Periods of sharp increases or decreases in specific channels.
# - If there are dips in one channel during a specific period, investigate external factors (e.g., marketing campaigns or   external disruptions like COVID-19).
# - Identify periods of high growth and replicate strategies that contributed to the success.

# In[13]:


plt.figure(figsize = (10, 6))
df_area.plot.line()


# ## Histogram
# 
# #### Insights
# 
# - The histogram likely shows how units sold are distributed across different intervals.
# - Peaks in specific bins indicate the most common range of units sold.
# - If the data skews towards lower values, strategies to increase purchase volume (e.g., bundle deals, discounts) could help.
# - High-frequency bins represent standard purchase volumes, and inventory planning should focus on these ranges.
# 

# In[14]:


plt.figure(figsize = (10, 6))

plt.hist(df['Units Sold'], bins = 10, color = 'skyblue', edgecolor = 'black')
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')


# ## Subplots
# 
# #### Insights
# 
# **Units Sold:**
# 
# - The distribution of units sold is right-skewed, indicating that most products sell a relatively small number of units, while a few products sell a large number of units.
# - The majority of products sell between 0 and 2,000 units.
# - There are a few outliers with very high sales volumes.
# 
# **Total Revenue:**
# 
# - The distribution of total revenue is also right-skewed, with most products generating lower revenue.
# - The majority of products generate revenue between 0 and 1 million dollars.
# - There are a few products with very high revenue.
# 
# 
# **Total Profit:**
# 
# -  distribution of total profit is right-skewed, with most products generating lower profit.
# - The majority of products generate profit between 0 and 0.5 million dollars.
# - There are a few products with very high profit.
# 
# 
# **Unit Cost:**
# 
# - The distribution of unit cost is right-skewed, with most products having a lower unit cost.
# - The majority of products have a unit cost between 0 and 100 dollars.
# - There are a few products with a higher unit cost.

# In[15]:


plt.figure(figsize = (10, 6))

plt.subplot(2, 2, 1)
plt.hist(df['Units Sold'], bins = 10, color = 'salmon', edgecolor = 'black')
plt.title('Frequency of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(df['Total Revenue'], bins = 10, color = 'lightcoral', edgecolor = 'black')
plt.title('Total Revenue Distribution')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(df['Total Profit'], bins = 10, color = 'lightgreen', edgecolor = 'black')
plt.title('Total Profit Distribution')
plt.xlabel('Total Profit')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(df['Unit Cost'], bins = 10, color = 'pink', edgecolor = 'black')
plt.title('Frequency of Unit Cost')
plt.xlabel('Unit Cost')
plt.ylabel('Frequency')

plt.tight_layout()


# ## Scatter plot
# 
# #### Insights
# 
# - There appears to be a positive correlation between price and units sold, suggesting that higher-priced products tend to have higher sales volumes.
# - However, the relationship is not perfectly linear, indicating that other factors might also influence sales.

# In[16]:


plt.figure(figsize = (10, 6))

plt.scatter(df['Unit Price'], df['Unit Cost'], color = 'skyblue', edgecolor = 'black')
plt.title('Unit Price VS Unit Cost')
plt.xlabel('Unit Price')
plt.ylabel('Unit Cost')


# ## Box plot
# 
# #### Insights
# 
# - The online channel shows a wider range of sales values compared to the offline channel.
# - The median sales value is higher for the online channel.
# - There are some outliers in the offline channel, indicating potentially unusual sales transactions.

# In[17]:


sns.boxplot(x = 'Sales Channel', y = 'Total Revenue', data = df)
plt.title('Total Revenue by Sales Channel')


# ## Bar Chart
# 
# #### Insights
# 
# - The online channel generates significantly higher total sales compared to the offline channel.
# - The difference in sales between the two channels is substantial.

# In[18]:


ax = sns.barplot(x = 'Sales Channel', y = 'Total Profit', data = df, palette = sns.color_palette('tab10'))

for bars in ax.containers:
    ax.bar_label(bars)
    
plt.show()


# ## Multivariate Analysis
# 
# #### Insights
# 
# - The line chart shows a general upward trend in total sales over time, indicating growth in the business.
# - There are fluctuations in the sales figures, possibly due to seasonal factors or marketing campaigns.
# - A sharp increase in sales can be observed around point "D", which might be attributed to a successful promotion or product launch.

# In[19]:


plt.figure(figsize = (10, 6))

sns.lineplot(x = 'Order Priority', y = 'Total Revenue', hue = 'Sales Channel', data = df, palette = 'tab10', marker = 'o')
plt.title('Relationship between Order Priority and Total Revenue')
plt.show()


# ## Pie chart
# 
# #### Insights
# 
# - The majority of sales come from the online channel.
# - The offline channel contributes a significant portion as well.
# - The distribution between the two channels is not evenly balanced.

# In[22]:


df_pie = df['Order Priority'].value_counts()

plt.figure(figsize = (6, 6))

plt.pie(df_pie, labels = df_pie.index, autopct = '%1.f%%', startangle = 90, colors = ['skyblue', 'pink', 'yellow', 'green'])
plt.title('Proportion of Transaction by Sales Channel')
plt.show()


# ## Bubble Chart
# 
# #### Insights
# 
# - There appears to be a positive correlation between price and units sold, suggesting that higher-priced products tend to have higher sales volumes.
# - The bubble size indicates profit, and there seems to be a cluster of bubbles with moderate profit levels.
# - A few larger bubbles suggest products with higher profit margins.

# In[21]:


plt.scatter(df['Units Sold'], df['Unit Price'], 
            s = df['Unit Cost'] / 5, alpha = 0.6, color = 'skyblue', edgecolor = 'black')

plt.title('Units Sold Bs Unit Price')
plt.xlabel('Units Sold')
plt.ylabel('Unit Price')
plt.tight_layout()

