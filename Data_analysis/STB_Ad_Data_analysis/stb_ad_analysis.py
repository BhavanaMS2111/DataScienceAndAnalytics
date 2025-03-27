import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.title("STB AD VIEWERSHIP ANALYSIS")

adInfo = pd.read_excel(r'D:\Data_analysis\STB_Ad_Data_analysis\ad_viewership_data.xlsx')
st.subheader("STB AD Details")
st.write(adInfo.head())
st.subheader("MISSING DATA FROM AD ANALYTICS DATASET")  
st.write(adInfo.isnull().sum())

for col in adInfo.select_dtypes(include='number').columns:
    adInfo[col] = adInfo[col].fillna(adInfo[col].mean())
for col in adInfo.select_dtypes(include='object').columns:
    adInfo[col] = adInfo[col].fillna(adInfo[col].mode()[0])

st.subheader("AD ANALYSTICS DATA AFTER MISSING VALUE TREATMENT")  
st.write(adInfo.head())

st.subheader("PERFORMING UNIVARIATE ANALYSIS")

for col in adInfo.select_dtypes(include='number').columns:
    fig,ax = plt.subplots()
    fig.set_figheight = 4
    fig.set_figheight = 4
    st.subheader(f'Histogram of {col}')
    sns.histplot(data=adInfo,x=col,ax=ax,bins=30,kde=True)
    st.pyplot(fig)

uni_hist_insights = """
For our dataset, the histogram can provide key insights for numerical variables like:

1. Total_Ad_Duration
A histogram of Total Ad Duration shows how much time ads are displayed across different STBs.
If the histogram is right-skewed, it means most ads have shorter durations, while some outliers have long durations.
If normally distributed, most ads have an average duration.
2. Watch_Duration
A histogram can help us see how long users watch ads.
If it’s right-skewed, it suggests most users watch for shorter durations, with only a few watching longer.
A bimodal distribution (two peaks) might indicate two types of viewers: short-duration watchers and long-duration watchers.
3. Ad_Click_Rate
A histogram can tell us whether most users engage with ads or ignore them.
If most values are close to zero, it means users don’t click on ads often.
A uniform distribution might suggest an even engagement across different ads.
4. Viewer_Age
Helps understand the age distribution of viewers.
If it’s bimodal, it may indicate two age groups watching ads (e.g., younger and older audiences).
If it’s right-skewed, most viewers are younger.
"""
st.subheader("Univariate Analysis using HISTOGRAM")
st.write(uni_hist_insights)


for col in adInfo.select_dtypes(include='number').columns:
    fig,ax = plt.subplots()
    fig.set_figheight = 4
    fig.set_figheight = 4
    st.subheader(f'Kernel Density Estimate of {col}')
    sns.kdeplot(data=adInfo,x=col,ax=ax,fill=True)
    st.pyplot(fig)
    
uni_kde_insights = """
Watch_Duration vs Total_Ad_Duration
If Watch_Duration has a peak around 3× Total_Ad_Duration, it confirms the expected pattern.
If the distribution is heavily right-skewed, some users may be watching for much longer than expected.
Ad_Click_Rate
A left-skewed distribution means most ads have very low click rates.
A bimodal distribution (two peaks) might indicate two different groups of users: one highly engaged and one uninterested.
"""
st.subheader("Univariate Analysis using Box Plot")
st.write(uni_kde_insights)

for col in adInfo.select_dtypes(include='number').columns:
    fig,ax = plt.subplots()
    fig.set_figheight = 4
    fig.set_figheight = 4
    st.subheader(f'Box Plot of {col}')
    sns.boxplot(y=adInfo[col],ax=ax)
    st.pyplot(fig)
    
uni_boxplot_insights = """
Watch_Duration & Total_Ad_Duration:

If the box is large → High variability in how long users watch ads.
If there are many outliers → Some users watch much longer/shorter than expected.

Ad_Click_Rate:

A small box means most click rates are similar.
Many outliers mean some ads have much higher/lower click rates than others.

You can see one outlier on the far right (near 5), which means some ads have significantly higher click rates than others.
This suggests a few ads are performing exceptionally well, while most ads have much lower click rates.

Possible Actions Based on This Analysis
Investigate High-Performing Ads (Outliers):

Why are these ads getting significantly higher clicks?
Is it due to better content, placement, or time of the day?
Can similar strategies be applied to other ads?
Handle the Skewness:

If the right skew is too extreme, consider log transformation or scaling methods for better modeling.
If the click rates are too low for many ads, investigate ad targeting, content, and engagement strategies.
"""
st.subheader("Univariate Analysis using BOX Plot")
st.write(uni_boxplot_insights)

for col in adInfo.select_dtypes(include='number').columns:
    fig,ax = plt.subplots()
    fig.set_figheight = 4
    fig.set_figheight = 4
    st.subheader(f'Violin Plot of {col}')
    sns.violinplot(y=adInfo[col],ax=ax)
    st.pyplot(fig)
    
uni_violin_plot_insights = """
1. Watch Duration
What to look for:
If the violin plot is symmetrical, it indicates a normal distribution.
If it is skewed to the right, it means more users watched for a shorter duration.
If it is skewed to the left, it means many users watched for a longer duration.
Thin regions indicate fewer data points, while wider regions indicate denser data.
Possible Inference:
If the violin plot shows right skew, it suggests that most viewers have shorter watch durations.
If it’s bimodal (two peaks), it suggests two distinct groups of viewers (e.g., those who skip ads and those who watch fully).
2. Total Ad Duration
What to look for:
If the distribution is tight and centered, it suggests a fixed ad duration.
If the distribution is wide, it suggests a large variation in ad durations.
Presence of outliers might indicate unusually long or short ads.
Possible Inference:
If there’s a clear peak at a certain duration, it suggests that most ads have a standardized length.
If the distribution is uniform, it suggests a wide variety of ad lengths.
3. Ad Click Rate
What to look for:
If the violin plot is skewed to the left, it suggests that most ads have low click-through rates.
If it has a fat tail, it suggests that a small percentage of ads perform exceptionally well.
Outliers might indicate viral or highly engaging ads.
Possible Inference:
If most click rates are concentrated at low values, it suggests that users are not frequently engaging with ads.
If there is a long right tail, it means a few ads perform exceptionally well in click-through rate.
Key Takeaways from Violin Plots
Identify Skewness: If the distribution is not symmetric, it tells us how the data is skewed.
Detect Outliers: Extreme values will be shown as separate points or elongated tails.
Understand Data Spread: A wider violin means a larger spread, while a narrow one means most values are concentrated.
Find Peaks & Modes: If the plot has multiple bulges, it suggests multiple groups within the data (e.g., different ad-watching behaviors).
"""
st.subheader("Univariate Analysis using VIOLIN Plot")
st.write(uni_violin_plot_insights)


st.subheader("UNIVARIATE ANALYSIS FOR CATEGORICAL DATA")

for col in adInfo.select_dtypes(include='object').columns:
    fig,ax = plt.subplots()
    fig.set_figheight = 4
    fig.set_figheight = 4
    st.subheader(f'COUNT Plot of {col}')
    sns.countplot(x=adInfo[col],ax=ax)
    st.pyplot(fig)
    
uni_count_plot_insights = """
Insights from Count Plots in This Dataset Context
1. Ad Provider
Shows which ad providers have the highest and lowest occurrences.
Helps identify dominant ad providers in the dataset.
2. Time of Day
Shows when most ads are being played (Morning, Afternoon, Evening, Night).
Helps analyze prime time for ad viewing.
3. Ad Position
Determines whether ads are mostly played at the start, middle, or end of content.
Useful for assessing which ad positions are most common.
4. Playback Type
Differentiates between Live TV vs. Recorded Playback.
Helps analyze whether users engage with ads more in live or recorded content.
5. STB Model
Shows which set-top box model is most frequently used.
Can help in device-specific ad optimization.
6. Device Type
Helps understand whether viewers are watching on TV, Mobile, Laptop, etc..
Useful for platform-based ad targeting.
7. Viewer Location
Identifies regional trends in ad viewership.
Helps advertisers customize ads for specific regions.
"""
st.subheader("Univariate Analysis using COUNT Plot")
st.write(uni_count_plot_insights)



for col in adInfo.select_dtypes(include='object').columns:
    fig, ax = plt.subplots(figsize=(6, 6))  # Define figure and size
    counts = adInfo[col].value_counts()  # Count unique values
    ax.pie(counts, labels=counts.index, colors=['lightblue', 'lightgreen', 'salmon','red','orange'])
    ax.set_title(f'Pie Plot of {col}')
    
    # Display in Streamlit
    st.subheader(f'PIE Plot of {col}')
    st.pyplot(fig)

st.subheader("BIVARIATE ANALYSIS")

st.title("SCATTER PLOT")
plt.figure(figsize=(6,4))
plt.title("")
fig,ax = plt.subplots()
sns.scatterplot(data=adInfo, x="Total_Ad_Duration", y="Watch_Duration")
plt.title("Scatter Plot: Total_Ad_Duration vs Watch_Duration")
st.pyplot(fig)
insights_scatter = """
Inference:

If points are aligned in a straight line, the correlation is strong.
If scattered randomly, there’s no correlation.
"""
st.write(insights_scatter)


st.title("JOINT PLOT")
plt.figure(figsize=(6,4))
plt.title("")
joint = sns.jointplot(data=adInfo, x="Total_Ad_Duration", y="Watch_Duration",kind='scatter')
plt.title("JOINT Plot: Total_Ad_Duration vs Watch_Duration")
st.pyplot(joint.figure)
insights_joint = """
Inference:

If points are aligned in a straight line, the correlation is strong.
If scattered randomly, there’s no correlation.
"""
st.write(insights_joint)


st.title("HEXBIN PLOT")
plt.figure(figsize=(6,4))
plt.title("")
joint = sns.jointplot(data=adInfo, x="Total_Ad_Duration", y="Watch_Duration",kind='hex')
plt.title("HEX BIN Plot: Total_Ad_Duration vs Watch_Duration")
st.pyplot(joint.figure)
insights_joint = """
Inference:

If points are aligned in a straight line, the correlation is strong.
If scattered randomly, there’s no correlation.
"""
st.write(insights_joint)

st.title("CORRELATION HEATMAP")
plt.figure(figsize=(6,4))
plt.title("")
numeric_data = adInfo.select_dtypes(include='number')
heat = sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("HEAT MAP Plot: Total_Ad_Duration vs Watch_Duration")
st.pyplot(heat.figure)


st.subheader("BIVARIATE ANALYSIS-- CATEGORICAL vs NUMERICAL DATA")
st.title("BOX PLOT")
plt.figure(figsize=(6,4))
box = sns.boxplot(data=adInfo, x="STB_Model", y="Watch_Duration")
plt.title("BOX Plot: STB_Model vs Watch_Duration")
st.pyplot(box.figure)

st.title("VIOLIN PLOT")
plt.figure(figsize=(6,4))
violin = sns.violinplot(data=adInfo, x="STB_Model", y="Watch_Duration")
plt.title("VIOLIN Plot: STB_Model vs Watch_Duration")
st.pyplot(violin.figure)


st.title("SWARM PLOT")
fig, ax = plt.subplots(figsize=(6,4))
sns.swarmplot(data=adInfo, x="STB_Model", y="Watch_Duration", size=3, ax=ax)
ax.set_title("SWARM Plot: STB_Model vs Watch_Duration")
st.pyplot(fig)



st.subheader("BIVARIATE ANALYSIS-- CATEGORICAL vs CATEGORICAL DATA")

st.title("HEATMAP PLOT")
fig, ax = plt.subplots(figsize=(6,4))
cross_tab = pd.crosstab(adInfo["STB_Model"], adInfo["Playback_Type"])
sns.heatmap(cross_tab, annot=True, cmap="Blues", fmt="d")
ax.set_title("HEAPMAP Plot: STB_Model vs Watch_Duration")
st.pyplot(fig)


st.title("CLUSTERED BAR CHART")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=adInfo, x="STB_Model", hue="Playback_Type")
plt.title("Clustered Bar Chart: STB Model vs Playback Type")
st.pyplot(fig)

st.title("MULTIVARIATE ANALYSIS-- PAIR PLOT")
num_vars = ["Ad Duration", "User Engagement", "Viewership"]
sns.pairplot(adInfo[num_vars], diag_kind="kde", plot_kws={'alpha': 0.6, 's': 80})
plt.suptitle("Pair Plot: Ad Duration, User Engagement, and Viewership", y=1.02)
plt.show()

st.title("MULTIVARIATE ANALYSIS-- SCATTER PLOT")
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(adInfo["Ad Duration"], adInfo["Viewership"], adInfo["User Engagement"], alpha=0.6, c=adInfo["User Engagement"], cmap='viridis')
# Labels and Title
ax.set_xlabel("Ad Duration (seconds)")
ax.set_ylabel("Viewership Count")
ax.set_zlabel("User Engagement Score")
ax.set_title("3D Scatter Plot: Ad Duration vs. Viewership vs. Engagement")
plt.show()

st.title("MULTIVARIATE ANALYSIS-- HEATMAP")
correlation_matrix = adInfo.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap: Numerical Variables in Ad Viewership Data")
plt.show()