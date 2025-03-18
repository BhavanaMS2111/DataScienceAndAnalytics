import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(30)

st.title("Sales Data Statistical Analysis")

data_set = {
    'product_id': range(1,21),
    'product_name': [f'Product {i}' for i in range(1,21)],
    'category': np.random.choice(['Clothing','Electronics','Home Decor','Sports','Lighting'],20),
    'units_sold': np.random.poisson(lam=20,size=20),
    'sale_date': pd.date_range(start='2025-03-17',periods=20,freq='D')
    
}
sales_data = pd.DataFrame(data_set);
st.header('Sales Data')
st.write(sales_data)

st.header('Checking Descriptive Statistics on the Data')
descriptive_stats = sales_data['units_sold'].describe().T
st.write(descriptive_stats)

st.header('Data Analysis Category Wise for the Sales Information.')
categ_data = sales_data.groupby('category')['units_sold'].agg(['sum','mean','std']).reset_index().rename(columns = {'category':'Category','sum':"Sum",'mean':'Mean','std':'Standard Deviation'})
st.write(categ_data)

st.header('Descriptive Statistics on Units Sold')
descriptive_stats_units_sold = {'Mean':sales_data['units_sold'].mean(),
                                'Median':sales_data['units_sold'].median(),
                                'Mode':sales_data['units_sold'].mode()[0],
                                'Variance':sales_data['units_sold'].var(),
                                'Standard Deviation':sales_data['units_sold'].std()}
desc_stat_units = pd.DataFrame(descriptive_stats_units_sold,index = [0])
st.write(desc_stat_units)

st.title('Performing Inferential Statistics')
confidence_level = .95
degrees_freedom = len(sales_data) - 1
sample_mean = sales_data['units_sold'].mean()
sample_standard_error = sales_data['units_sold'].std() / np.sqrt(len(sales_data['units_sold']))

#t score for 95 % confidence level is
t_score = stats.t.ppf((1+confidence_level)/2,degrees_freedom)
margin_of_error = t_score * sample_standard_error

confidence_interval =  (sample_mean - margin_of_error, sample_mean + margin_of_error)
st.subheader("Confidence Interval for the Mean of Units Sold Considering 95% Confidence:")
st.write(confidence_interval)


confidence_level = .99

#t score for 99 % confidence level is
t_score = stats.t.ppf((1+confidence_level)/2,degrees_freedom)
margin_of_error = t_score * sample_standard_error

confidence_interval =  (sample_mean - margin_of_error, sample_mean + margin_of_error)
st.subheader("Confidence Interval for the Mean of Units Sold Considering 99% Confidence:")
st.write(confidence_interval)


# Null Hypothesis:- Mean of the Units Sold is 20
# Alternate Hypothesis:- Mean of the Units sold is not equal to 20

st.write('Performing t-test as sample is small size')
t_statistic,p_value = stats.ttest_1samp(sales_data['units_sold'],20)

st.write(f"t_statistic : {t_statistic}  p_value : {p_value}")

if (p_value < 0.05) :
    st.write('Reject Null Hypothesis. Mean Units Sold is not equal to 20')
else :
    st.write('Accept Null Hypothesis. Mean Units Sold is equal to 20')


st.subheader('Dist Plot')
sns.set_style('whitegrid')

plt.figure(figsize=(10,6))
sns.histplot(sales_data['units_sold'], bins=20, kde=True)
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.axvline(sales_data['units_sold'].mean(),color='r',linestyle='--',label='Mean')
plt.axvline(sales_data['units_sold'].median(),color='g',linestyle='--',label='Median')
plt.axvline(sales_data['units_sold'].mode()[0],color='b',linestyle='--',label='Mode')
plt.legend()
st.pyplot(plt.gcf())

plt.figure(figsize=(10,6))
plt.title('Box Plot for Units Sold per Category')
plt.xlabel('Category')
plt.ylabel('Units Sold')
sns.boxplot(x=sales_data['category'], y = sales_data['units_sold'])
st.pyplot(plt.gcf())


plt.figure(figsize=(10,6))
plt.title('Box Plot for Units Sold per Category')
plt.xlabel('Category')
plt.ylabel('Units Sold')
sns.barplot(x='Category', y='Sum', data=categ_data)
st.pyplot(plt.gcf())