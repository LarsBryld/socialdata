#!/usr/bin/env python
# coding: utf-8

# #### Project guidelines (comment to be removed)
# 
# https://github.com/suneman/socialdata2021/wiki/Final-Project

# # Motivation.
# 
# 
# ### What is your dataset?
# * The dataset contains traffic hourly data on some of the busiest roads in Copenhagen collected between 2005 and 2014.
# * the data have been downloaded from this open data page: 
# https://www.opendata.dk/city-of-copenhagen/faste-trafiktaellinger#resource-faste_trafikt%C3%A6llinger_2008.xlsx
# *  Each year of data has been merged into one dataset that we are using for our analysis.
# 
# 
# * The weather data have been downloaded from the Danish Meteorological Institute
# https://www.dmi.dk/vejrarkiv/
# 
# * Copenhagen GeoJson polygons (districts maps) instead have been downloaded from here
# https://giedriusk.carto.com/tables/copenhagen_districts/public
# 
# 
# ### Why did you choose this/these particular dataset(s)?
# - opt 1) 
#     - describing CPH traffic flows over time and space 
#     - identify patterns in the data that allow for classification of traffic volumes by roads 
# - opt 2) merging Copenhagen traffic data with weather data to predict traffic volumes over time and space
# 
# ### What was your goal for the end user's experience?
#     Easily explaine how we cleaned the data and why.
# Building a tool that allows for an easy visualizion of traffic volumes/flows across time and space 
# 
# 
# (...............and weather conditions? maybe)
# 
# 
# 
# 
# # Basic stats. Let's understand the dataset better
# 
# ### Dataset stats
# 
# The dataset has **183k rows**, with **30 columns** and after cleaning, preprocessing and transformation of the data it has increased to 1.4 million rows. 
# The original columns contain the following information:
#    - Vej-Id, these markers end with values "T", "+" or "-":
#         - "T" are Total vehicles detected on each hour/road/detection point;
#         - "+" are the vehicles moving in the direction of increasing house numbers: "+" means that the house numbers go up (1,2,3....). We assumed that the roads numbering starts from the city center and increases with the distance. So intuitively these vehicles should be the ones leaving the city. This is often true, although not always, as confirmed by the CPH traffic data owners, but we will show in our visualizations that it holds for most roads;
#         - "-" should then be the vehicles entering the city (vehicles that physically move "against" the house numbering);
#    - Vejnavn contains the Road Names;
#    - UTM geographical coordinates of the traffic detection points;
#    - Date of detection;
#    - Hour of detection: basically there are 24 traffic columns for each date row;
#   
# ### Our choices in data cleaning, preprocessing and data transformation
#   
# The original data have been transformed in the following ways:
#    - Vej-Id data have been manipulated to isolate only the last element: "T", "+", "-";
#    - UTM coordinates have been transformed into Latitude and Logitude coordinates;
#    - hourly column data have been convereted into rows. This has increased the number of rows by a factor of 24, to around 4.4m rows;
#    - Vej-Id markers have been used to move "Leaving" and "Entering" traffic data from columns to rows. This has allowed us to create 3 new features: Leaving vehichles, Entering vehichles and "Net Traffic Flow" data, that are very useful in showing hourly traffic patterns and for our ML classiffication tool. This transformation has of course reduced the number of rows to 1/3 to around 1.45m rows;
#    - we have randomized Latitude and Longitude, and added these data in 2 new columns ("Lat_rand", "Lon_rand"). this was necessary to facilitate spacial data visualizations;  
#    - Finally we have created new features to visualize the data by different timeframes: daily, weekly, yearly, etc.
#    - As part of our data preprocessing we have also deleted all the empty or otherwise irrelavant columns of data
# 
# 
# 

# In[ ]:


#import geopandas as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import utm
import folium
import json


# Downloading data and removing/adding features

# In[ ]:


# Downloading faste-trafiktaellinger-2008_clean (to be changed if we get data directly form website)
#df = pd.read_csv("C:/Users/User/Dropbox/DTU/02806 Social data analysis and visualization/cph_traffic_2005-2014_original.csv",
#                 parse_dates = ['Dato'],encoding='ISO-8859-1')

df1 = pd.read_csv("https://raw.githubusercontent.com/LarsBryld/socialdata/main/cph_traffic_2005-2009_original.csv",
                 parse_dates = ['Dato'],encoding='ISO-8859-1')
df2 = pd.read_csv("https://raw.githubusercontent.com/LarsBryld/socialdata/main/cph_traffic_2010-2014_original.csv",
                 parse_dates = ['Dato'],encoding='ISO-8859-1')

df = pd.concat((df1,df2))

# cleaning Vej-Id for more clear traffic directions
df['Vej-Id'] = df['Vej-Id'].str.split(n=4).str[-1]

#change the Hours headers
for i in range(7,31):
    df = df.rename(columns={df.columns[i]: df.columns[i].split('.')[1].split('-')[0]})
    df[df.columns[i]] = df[df.columns[i]].str.replace(',', '').fillna(0).astype('float')

### converting UTM coordinates into Latitute/Longitude using the "utm" library: https://pypi.org/project/utm/
# first we create a function that applies the utm api to 2 Series of data
def uf(x):
    return utm.to_latlon(x[0], x[1], 32, 'T')
# then we apply this function to the UTM coordinates in the file
#df['LatLon'] = df[['Easting','Northing']].apply(uf, axis=1)
df[['Lat', 'Lon']] = pd.DataFrame(df[['(UTM32)','(UTM32).1']].apply(uf, axis=1).tolist(), index=df.index)

# removing the unwanted columns
df = df.drop(columns = ['Unnamed: 0','Spor','(UTM32)','(UTM32).1'])

# converting hours data columns into rows
df = df.melt(id_vars=["Vej-Id","Vejnavn","Dato","Lat","Lon"],
        value_vars=['00','01','02','03','04','05','06','07','08','09','10','11','12',
                    '13', '14','15', '16','17', '18','19','20','21','22','23'],
        var_name="Hour", 
        value_name="Vehicles")

### moving rows data for Vehicles Entering the City, Leaving the City and Net Traffic Flows into columns
# Selecting only Entering Vehicles (and creating a unique index)
df_ent = df[df['Vej-Id'] == '-']
df_ent['index'] = df_ent['Vejnavn'] + df_ent['Dato'].dt.strftime('%Y-%m-%d') + df_ent['Hour']
df_ent = df_ent.set_index('index')
# Selecting only Leaving Vehicles (and creating a unique index)
df_ex = df[df['Vej-Id'] == '+']
df_ex['index'] = df_ex['Vejnavn'] + df_ex['Dato'].dt.strftime('%Y-%m-%d') + df_ex['Hour']
df_ex = df_ex.set_index('index')
# Selecting only Total Vehicles on the roads (and creating a unique index)
df = df[df['Vej-Id'] == 'T']
df['index'] = df['Vejnavn'] + df['Dato'].dt.strftime('%Y-%m-%d') + df['Hour']
df = df.set_index('index')
# adding columns for Vehicles Entering the City, Leaving and Net Traffic Flows
df['Entering Vehicles'] = df_ent['Vehicles']
df['Leaving Vehicles'] = df_ex['Vehicles']
df['Net Traffic Flow'] = df['Entering Vehicles'] - df['Leaving Vehicles']
# renaming the Total Vehicles column
df = df.rename(columns={"Vehicles": "Total Vehicles"})

# randomizing Latitude and longitude points
mu, sigma1 = 0, 0.0015
mu, sigma2 = 0, 0.003
noise1 = np.random.normal(mu, sigma1, [len(df),1])
noise2 = np.random.normal(mu, sigma2, [len(df),1]) 
df[['Lat_rand']] = df[['Lat']] + noise1
df[['Lon_rand']] = df[['Lon']] + noise2

# Add Day of the Week, Day, ,Week, Month, Year,
df["DayName"] = df['Dato'].apply(lambda x: x.day_name())
df["WeekDay"] = df['Dato'].dt.weekday
df["DayOfMonth"] = df['Dato'].dt.day
df["Week"] = df['Dato'].dt.week
df["Month"] = df['Dato'].dt.month
df["Year"] = df['Dato'].dt.year

# removing the 'Vej-Id' columns to avoid confusion (now all traffic data are in the columns for "-" , "+" and "T")
df = df.drop(columns = ['Vej-Id'])

#df.head(10)


# ## Key points/plots from our exploratory data analysis
# 
# ### Total traffic distribution by Road
# 
# 
# Below is the count of (total) vehicles per each Road in descending order recorded in the whole period.
# From both the list and the plot below you can see how there are significant differences in traffic volumes among the roads available in the dataset

# In[ ]:


totcount = df.groupby('Vejnavn')['Total Vehicles'].sum().sort_values(ascending=False)
pd.DataFrame(totcount.values, index = list(totcount.index), columns =['Total Vehicles']) 


# In[ ]:


pd.DataFrame(totcount.values, index = list(totcount.index), columns =['Vehicles']).plot.bar()


# ## Traffic distribution over time

# 
# ### Monthly distribution of total vehicles (All roads)
# 
# The main pattern observable is the drop in traffic in July and December: Danish holiday season

# In[ ]:


df.groupby('Month')['Total Vehicles'].sum().plot.bar()


# ### Montly distribution per Road (Total vehicles)
# * The main feature that is clearly visible is the drop in taffic on nearly all roads on July, which is the month where most Copenhageners are on holidays away from the city
# * a few roads that except these rule show unclear patterns that could be due to data quality issues (check if this is true after wi enclose other years

# In[ ]:


m = df.groupby(["Month", "Vejnavn"]).sum()["Total Vehicles"].unstack()
#m


# In[ ]:


m.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# 
# ### Weekly distribution of total vehicles (All roads)
# 
# The main pattern observable is the w-e drop in traffic

# In[ ]:


df.groupby('WeekDay')['Total Vehicles'].sum().plot.bar()


# ### Weekly distribution per Road (Total vehicles)
# From the plots below a clear drop of traffic on all roads is clearly visible.
# 2 main exceptions to this pattern are:
# * **Kalvebod Brygge** where the drop happens on Mondays and tuesdays. Althought this could be due to some quality issue about the data (we need to check if this is still the case when we include all years (now we are only working with 2008 data)
# * **Jagtvej** shows a much lower drop in the w-e compared to other roads
# 

# In[ ]:


w = df.groupby(["WeekDay", "Vejnavn"]).sum()["Total Vehicles"].unstack()
#w


# In[ ]:


w.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# 
# ### Day of Month distribution of total vehicles (All roads)
# 
# one pattern that can be observed is that the 31st day of the month shows a little more than half the volumes of the average of the other days. This is probably due to the fact that there are ony 7 months that contain 31 days

# In[ ]:


df.groupby('DayOfMonth')['Total Vehicles'].sum().plot.bar()


# ### Day of the month distribution per Road (Total vehicles)
# No patterns clearly visible across days of the month. The only exception os of course that the total traffic on the 31st day of the month is around half of the other days, but thisis of course due to the fact that only 7 months over 12 have 31 days

# In[ ]:


d = df.groupby(["DayOfMonth", "Vejnavn"]).sum()["Total Vehicles"].unstack()
#d


# In[ ]:


d.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# ### Hourly distribution per Road (Total vehicles)
# **Nearly all roads share the same pattern in hourly traffic flows:**
# - Midnight to 5am: very low traffic 
# - 6-8am: people go to work and vehicles volumes increase rapidly for 3 hours. Then a little slow down for a couple of hours
# - 11 to 15-16: traffic volumes start growing again until they peak when people start going bach home from work. 
# - 17-18: vehicles numbers drop consistently. Dinner time in Denmark
# - 19-23: the traffic flows slowly reduce

# In[ ]:


h = df.groupby(["Hour", "Vejnavn"]).sum()["Total Vehicles"].unstack()
#h


# In[ ]:


h.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# ### Hourly distribution per Road (Net Entry-Exit flows)
# **REMEMBER: the Exit-Entry analysis is based on one big assumptions: roads numbering follows an ascending order that starts at zero from the City center and then increases with the distance from the City center
# 
# **Most roads share the same pattern shapes, but not all:**
# - Midnight to 4am: most roads show cars exiting the city
# - 5-9am: vehicles entering the city are the majority and inward volumes constantly increase until they peak around 9-10
# - 11 to 15-16: inward traffic volumes start dwindling until the outwarding vehicles start taking over from 15
# - 15-18: majority of vehicles are the one exiting the city
# - 19-23: no clear pattern: some roads show the majority of vehicles enterring the city again, while others show the majority of cars exiting the city, depending on the timeframe
# - some roads, like Roskildevej and Torvegade show opposite patterns than the one described above. The reason is probably because these roads lead to specific locations that attract a high number of workers, respectively **Roskilde and Amager**
# - some other roads instead show systematically net inward or systematic net outward flow of vehicles during the whole day. These are, respectively:
#   - .... (inward flow)
#   - .... (ouward flow)

# In[ ]:


hd = df.groupby(["Hour", "Vejnavn"]).sum()["Net Traffic Flow"].unstack()
#hd


# In[ ]:


hd.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# # Visualization of traffic volumes for each CPH district over time
# 
# * first we find CPH districts through a GeoJson file
# * then we map CPH roads to the relevant districts 
# * then we add the distric information to the main DataFrame
# * then we group the district data by timeframe: weekly/hourly seem like interesting timeframes to investigate
# * finally we represent the data on the interactive Choropleth Map

# In[ ]:


# first we upload CPH districts polygons from GeoJson file
import urllib.request, json 

with urllib.request.urlopen("https://raw.githubusercontent.com/LarsBryld/socialdata/main/copenhagen_districts.geojson") as url:
    cph_districts = json.loads(url.read().decode())

# we extract the districts names from the GeoJson file
districts = []
for i in range(len(cph_districts["features"])):
    districts.append(cph_districts["features"][i]['properties']['name'])
    
#districts


# In[ ]:


# then we create a unique list of CPH roads, with corresponding Longitude and Latitute
dfu = pd.concat((pd.DataFrame(df['Vejnavn'].unique(),columns=['Vejnavn']),
           pd.DataFrame(df['Lon'].unique(),columns=['Lon']),
           pd.DataFrame(df['Lat'].unique(),columns=['Lat'])), axis=1)
#dfu


# In[ ]:


# finally we find in what district each road falls, using the code from this example:
# https://stackoverflow.com/questions/57727739/how-to-determine-if-a-point-is-inside-a-polygon-using-geojson-and-shapely

from shapely.geometry import shape, GeometryCollection, Point

dist = []

for i in range(len(dfu)):
    for j in range(len(districts)):
        if shape(cph_districts["features"][j]['geometry']).contains(Point(dfu['Lon'][i],dfu['Lat'][i])):
            dist.append(cph_districts["features"][j]['properties']['name'])


# In[ ]:


# then we add the district name to our list of unique roads
dfu = pd.concat((dfu, pd.DataFrame(dist,columns=['District'])), axis=1)
dfu


# In[ ]:


# adding the District info to the original DF
df = pd.merge(df, dfu[['Vejnavn','District']], on="Vejnavn")
df


# In[ ]:


# grouping traffic data by Year and District
dfc = df.groupby(["Year", "District"]).mean()["Total Vehicles"].unstack()
dfc = dfc.reset_index()
dfc


# In[ ]:


# for the Plotly Choropleth to work we nee to put all columns into rows 

dfc = dfc.melt(id_vars=['Year'],
        value_vars=dfc.columns.values[1:],
        var_name="District", 
        value_name="Total Vehicles")
dfc


# In[ ]:


# creating an interactive map for CPH traffic
import plotly.express as px

max_value = dfc['Total Vehicles'].max()
fig = px.choropleth(dfc, locations='District',
                    geojson=cph_districts, featureidkey="properties.name",
                           color='Total Vehicles',
                           color_continuous_scale="Viridis",
                           range_color=(0, max_value),
                           projection="mercator",
                    animation_frame="Year", animation_group="District"
                          )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


# grouping traffic data by Hour and District
dfh = df.groupby(["Hour", "District"]).mean()["Total Vehicles"].unstack()
dfh = dfh.reset_index()
dfh


# In[ ]:


# for the Plotly Choropleth to work we nee to put all columns into rows 

dfh = dfh.melt(id_vars=['Hour'],
        value_vars=dfh.columns.values[1:],
        var_name="District", 
        value_name="Total Vehicles")
dfh


# In[ ]:


# creating an interactive map for CPH traffic
import plotly.express as px

max_value = dfh['Total Vehicles'].max()
min_value = dfh['Total Vehicles'].min()
fig = px.choropleth(dfh, locations='District',
                    geojson=cph_districts, featureidkey="properties.name",
                           color='Total Vehicles',
                           color_continuous_scale="Viridis",
                           range_color=(min_value, max_value),
                           projection="mercator",
                    animation_frame="Hour", animation_group="District"
                          )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Data Analysis
# ### Describe your data analysis and explain what you've learned about the dataset.
# Our data visualizations have shown us that:
# - there are huge differences in traffic across different roads in Copenhagen. This is of course very intuitive because different roads serve differennt purposes: the big traffic arteries have been build with the aim of carrying most of the daily traffic inside and outside of the city, while smaller ones are only used for local traffic. A good example of 2 roads like these are, for example, Ellebjergvej and Mozartsvej. These 2 roads are very close to each other but they serve completely different purposes. In fact: 
#     - Ellebjergvej allows people to get in and out of the city and is one of the busiest roads in our dataset, while 
#     - Mozartsvej only serves the local traffic and is the second less busy road in the dataset.
# - Monthly traffic is pretty stable across all months, with 2 biggest exceptions during Danish the holidays, when the traffic slows down compared to the other months: July and December
# - Weekly traffic in Copenhagen decreases in the weekend, as expected, for all roads
# - Hourly (total) traffic shows a pretty consistent pattern across all roads: very low traffic during the night; increase in the early hours of the day (6-8am) when people go to work. It keeps growing until it reaches a peak around 15-16 when people start going back home from work. Beween 16-18 the traffic quickly decreases, and after dinner time (19 it slows down a lot.
# - hourly net flows also show a pretty interesting and consistent pattern: 
#     - for most roads the net traffic flow outside of the city during the night; 
#     - the majority of vehicles starts flowing to the city from 4am, and the process continues until it peaks around 10-11: people leaving outside of the city go to their workplaces or to businesses that are located inside the city; 
#     - after 11am the net inflow of cars starts going down and 
#     - around 15 the majority of vehicles on the roads are the ones leaving the city: the end of the working day starts approaching.
#     - there is only one big exception to the above net traffic flow: Roskildevej shows a pattern opposite to all other roads. This probably suggests that the number of Copehageners working in Roskilde is higher that the number of people living in Roskilde and working in CPH 
#     - a handful of other roads **(......)** don't show the hourly net flow of traffic that we have just destribed, and instead these show one-sided traffic flow across the whole day. This is probably due to the fact that our assumption about identifying the traffic direction with the Vej-Id doesn't hold for these roads.  
# 
# 
# ### If relevant, talk about your machine-learning
# Of course the 

# # Machine Learning for classifying traffic data by Roads
# 
# The visualization of Hourly Net Flows (vehicles leaving - vehicles entering the city) above shows that, for example, Roskildevej has a very different pattern from other roads: cars on Roskildevej are leaving the city in the morning and are coming back at night. This is probably due to the fact that the number of Copenhageners working in Roskilde is higher than the number of Roskilde residents working in Copenhagen.
# 
# Based on this visual information we have build a classifier that can identify traffic flows on Roskildevej from roads that have a completely different hourly traffic flow (we have chosen Ellebjergvej for our example).
# 
# Our classifier uses Random Forests and yields around 80% accurate predictions for both the training and test data
# 

# In[ ]:


# Test without Weather data

### data preprocessing

# selecting our sample: focussing on data for 2011, 2012, 2013 and 2 roads only
dfml = df[(df['Year'].isin([2011,2012,2013]))
         & (df['Vejnavn'].isin(['Ellebjergvej','Roskildevej']))]  

# keeping only: Hour-of-the-day, Day-of-the-week, Month-of-the-year, and PD-District
dfml = dfml[['Vejnavn','Net Traffic Flow', 'Hour']]
#dfml


# In[ ]:


# encoding data with LabelEncoder
from sklearn.preprocessing import LabelEncoder

#creating labelencoder
le = LabelEncoder()
# encoding string labels (Category is the target variable)
labels = le.fit_transform(dfml['Vejnavn'])

# encoding string features (not necessary)
#NetTraffic =le.fit_transform(dfml['Hour'])

features=dfml[['Net Traffic Flow', 'Hour']]

#Split Train/test datasets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(X_train,  y_train)

# measuring the classification performance of the RF classifier through cross_val_score
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf, X_train,  y_train, cv=10))


# In[ ]:


# Measuring RF prediction performance (in the Test sample)
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test) # 0:Overcast, 2:Mild
print(classification_report(y_test, y_pred))#, target_names=target_names))


# # Machine Learing to predict traffic volumes with weather data

# ### Importing CPH weather data

# In[ ]:


dfw = pd.read_csv("https://raw.githubusercontent.com/LarsBryld/socialdata/main/cph_weather.csv",
                 parse_dates = ['DateTime'])
dfw


# In[ ]:


dfm = pd.merge(df, dfw, left_on='Dato',right_on='DateTime')
dfm


# In[ ]:


dfml = dfm[#(dfm['Vejnavn'] == 'Ellebjergvej') &
           (dfm['WeekDay']<=4) &
           (~dfm['Month'].isin([1,7,12]))].groupby(['Dato']).mean()

dfml


# In[ ]:


dfml['Total Vehicles'].plot.hist(bins=3, alpha=0.5)


# In[ ]:


pd.cut(dfml['Total Vehicles'], bins=3, retbins=True, labels=False)[0]


# In[ ]:


#creating labes by splitting floating data into bins
labels = pd.cut(dfml['Total Vehicles'], bins=3, retbins=True, labels=False)[0]

# encoding string features is not necessary
features = dfml[["LowTemp",
               "HighTem",
               "MidTemp",
               "AirPressure",
               "Rain",
               "LowWind",
               "MidWind",
               "HighWind",
               "Sunshine"]]

#Split Train/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(X_train,  y_train)



# measuring the classification performance of the RF classifier through cross_val_score
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf, X_train,  y_train, cv=10))


# In[ ]:


# Measuring RF prediction performance (in the Test sample)
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test) # 0:Overcast, 2:Mild
print(classification_report(y_test, y_pred))#, target_names=target_names))


# ### the good Classifier's results are probably driven by a class imbalance: class 1 represents 80%+ of the data when we look at the histogram that represents the data distribution.
# 
# ### so we need to increase the sample size of the other 2 classes and then see if the classifier has still such a high classification power.
# 
# ### how do we do it? first we add the labels to our dataframe

# In[ ]:


labels


# In[ ]:


dfml['Labels'] = labels
dfml#['Labels']


# ### then we split the dataframe by each label and we make the 3 categories (labels) into the same size 

# In[ ]:


# setting Max Class Size equal to 1.000
mcs = 1000


# extracting random samples of equally sized classes
dfs = pd.DataFrame()
dfs = dfs.append(dfml[dfml['Labels'] == 0].sample(n=mcs, random_state=1, replace=True))
dfs = dfs.append(dfml[dfml['Labels'] == 1].sample(n=mcs, random_state=1, replace=True))
dfs = dfs.append(dfml[dfml['Labels'] == 2].sample(n=mcs, random_state=1, replace=True)) 

dfs


# In[ ]:


dfs['Labels'].plot.hist(bins=5, alpha=0.5)


# In[ ]:


#taking the labels from the df
labels = dfs['Labels']

# tafing weather features from the df
features = dfs[["LowTemp",
               "HighTem",
               "MidTemp",
               "AirPressure",
               "Rain",
               "LowWind",
               "MidWind",
               "HighWind",
               "Sunshine"]]

#Split Train/test datasets
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(features, labels, test_size=0.33, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf1 = clf1.fit(X_train1,  y_train1)



# measuring the classification performance of the RF classifier through cross_val_score
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf1, X_train1,  y_train1, cv=10))


# In[ ]:


# Measuring RF prediction performance (in the Test sample)
from sklearn.metrics import classification_report

y_pred1 = clf1.predict(X_test1) # 0:Overcast, 2:Mild
print(classification_report(y_test1, y_pred1))#, target_names=target_names))


# ### let's try the fitted model on the original data

# In[ ]:


# taking features and labels from original dataframe
#taking the labels from the df
labels = dfml['Labels']

# tafing weather features from the df
features = dfml[["LowTemp",
               "HighTem",
               "MidTemp",
               "AirPressure",
               "Rain",
               "LowWind",
               "MidWind",
               "HighWind",
               "Sunshine"]]

# Measuring RF prediction performance (in the Test sample)
from sklearn.metrics import classification_report

y_pred = clf.predict(features) # 0:Overcast, 2:Mild
print(classification_report(labels, y_pred))#, target_names=target_names))


# ### lets' try to fit the model on a new road: Torvegade.

# In[ ]:


dfn = dfm[(dfm['Vejnavn'] == 'Torvegade') 
           & (dfm['WeekDay']<=4)
           & (~dfm['Month'].isin([1,7,12]))].groupby(['Dato']).mean()

dfn


# In[ ]:


pd.cut(dfn['Total Vehicles'], bins=3, retbins=True, labels=False)[0]


# In[ ]:


dfn['Total Vehicles'].plot.hist(bins=3, alpha=0.5)


# In[ ]:


#creating labes by splitting floating data into bins
labels = pd.cut(dfn['Total Vehicles'], bins=3, retbins=True, labels=False)[0]

# encoding string features is not necessary
features = dfn[["LowTemp",
               "HighTem",
               "MidTemp",
               "AirPressure",
               "Rain",
               "LowWind",
               "MidWind",
               "HighWind",
               "Sunshine"]]

#Split Train/test datasets
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(features, labels, test_size=0.33, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf2 = clf2.fit(X_train2,  y_train2)



# measuring the classification performance of the RF classifier through cross_val_score
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf2,features,  labels, cv=10))

# Measuring RF prediction performance (in the Test sample)
from sklearn.metrics import classification_report

y_pred2 = clf2.predict(features) # 0:Overcast, 2:Mild
print(classification_report(labels, y_pred2))#, target_names=target_names))


# In[ ]:


labels


# In[ ]:


dfn['Labels'] = labels


# In[ ]:


# setting Max Class Size equal to 1.000
mcs = 1000


# extracting random samples of equally sized classes
dfns = pd.DataFrame()
dfns = dfns.append(dfn[dfn['Labels'] == 0].sample(n=mcs, random_state=1, replace=True))
dfns = dfns.append(dfn[dfn['Labels'] == 1].sample(n=mcs, random_state=1, replace=True))
dfns = dfns.append(dfn[dfn['Labels'] == 2].sample(n=mcs, random_state=1, replace=True)) 

dfns


# In[ ]:


dfns['Labels'].plot.hist(bins=5, alpha=0.5)


# In[ ]:


#taking the labels from the df
labels = dfns['Labels']

# tafing weather features from the df
features = dfns[["LowTemp",
               "HighTem",
               "MidTemp",
               "AirPressure",
               "Rain",
               "LowWind",
               "MidWind",
               "HighWind",
               "Sunshine"]]

#Split Train/test datasets
from sklearn.model_selection import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(features, labels, test_size=0.33, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf3 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf3 = clf3.fit(X_train3,  y_train3)



# measuring the classification performance of the RF classifier through cross_val_score
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf3, X_train3,  y_train3, cv=10))


# ### Let's find which weather data are driving the classification

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(clf1, random_state=1).fit(X_test1, y_test1)
eli5.show_weights(perm, feature_names = X_test1.columns.tolist())


# In[ ]:


perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(clf1, random_state=1).fit(X_test1, y_test1)
eli5.show_weights(perm, feature_names = X_test1.columns.tolist())


# In[ ]:


perm = PermutationImportance(clf2, random_state=1).fit(X_test2, y_test2)
eli5.show_weights(perm, feature_names = X_test2.columns.tolist())


# In[ ]:


perm = PermutationImportance(clf3, random_state=1).fit(X_test3, y_test3)
eli5.show_weights(perm, feature_names = X_test3.columns.tolist())


# # Visualize data in space/time to identify some patterns

# # Radhusplads

# In[ ]:


import folium

map_hooray = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12.5)

folium.Marker([55.6761, 12.5683], 
              popup='RadHus Plads', 
              icon=folium.Icon(color='blue')
             ).add_to(map_hooray)

map_hooray


# # Visualizing some traffic data (randomized locations)

# In[ ]:


df1 = df[(df['DayOfMonth'].isin([1,2,3,4,5,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20,
                                21,22,23,24,25,26,27,28,29,30,31]))
         & (df['Hour'].isin(['07','08']))
         & (df['Month'].isin([6, 7]))
         & (df['Year'] == 2012)]

df1


# In[ ]:


map2 = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

folium.Marker([55.6761, 12.5683], 
              popup='City Hall', 
              icon=folium.Icon(color='blue')
             ).add_to(map2)

for i in range(len(df1)):
    folium.Circle(location=[df1.iloc[i]['Lat_rand'], df1.iloc[i]['Lon_rand']],
                  popup=df1.iloc[i]['Month'],
                  radius=4, #data.iloc[i]['value']*10000,
                  color='crimson',
                  fill=True,
                  fill_color='crimson'
                 ).add_to(map2)

map2


# # Heatmap

# In[ ]:


from folium.plugins import HeatMap

map_hooray = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = df1[['Lat', 'Lon']]
#heat_df = heat_df.dropna(axis=0, subset=['Y','X'])



# List comprehension to make out list of lists
heat_data = [[row['Lat'],row['Lon']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(map_hooray)

# Display the map
map_hooray


# # HeatMapWithTime  (Weekdays - NON random locations) 

# In[ ]:


from folium import plugins

map_hooray = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = df1[['Lat', 'Lon','Lat_rand', 'Lon_rand']]
#heat_df = heat_df.dropna(axis=0, subset=['Y','X'])

# List comprehension to make out list of lists
heat_data = [[row['Lat'],row['Lon']] for index, row in heat_df.iterrows()]

# Create weight column, using date
heat_df['Weight'] = df1['WeekDay']
heat_df['Weight'] = heat_df['Weight'].astype(float)
heat_df = heat_df.dropna(axis=0, subset=['Lat','Lon', 'Weight'])

# List comprehension to make out list of lists
heat_data = [[[row['Lat'],row['Lon']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(0,7)]

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(map_hooray)
# Display the map
map_hooray


# # HeatMapWithTime  (Weekdays - randomized locations)

# In[ ]:


from folium import plugins

map_hooray = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

heat_rn = pd.concat([heat_df['Lat_rand'], heat_df['Lon_rand'], heat_df['Weight']], axis=1)

# List comprehension to make out list of lists
heat_data_rn = [[[row['Lat_rand'],row['Lon_rand']] for index, row in heat_rn[heat_rn['Weight'] == i].iterrows()] for i in range(0,7)]

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data_rn,auto_play=True,max_opacity=0.8)
hm.add_to(map_hooray)
# Display the map
map_hooray


# # HeatMapWithTime  (Hours - only 2 roads) - we could skip this one

# In[ ]:


df3 = df[(df['DayOfMonth'].isin([1,2,3,4,5,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20,
                                21,22,23,24,25,26,27,28,29,30,31]))
         & (df['Vejnavn'].isin(['Roskildevej','Ellebjergvej']))
#         & (df['Hour'].isin(['07','08']))
         & (df['Month'].isin([6, 7]))
         & (df['Year'] == 2012)]
#        & (df['Vej-Id'] == 'T')]

#df3


# In[ ]:


map3 = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

folium.Marker([55.6761, 12.5683], 
              popup='City Hall', 
              icon=folium.Icon(color='blue')
             ).add_to(map3)

for i in range(len(df3)):
    folium.Circle(location=[df3.iloc[i]['Lat_rand'], df3.iloc[i]['Lon_rand']],
                  popup=df3.iloc[i]['Month'],
                  radius=4, #data.iloc[i]['value']*10000,
                  color='crimson',
                  fill=True,
                  fill_color='crimson'
                 ).add_to(map3)

map3


# In[ ]:


from folium import plugins

map_hooray = folium.Map([55.6761, 12.5683], tiles = "Stamen Toner", zoom_start=12)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = df3[['Lat_rand', 'Lon_rand']]
#heat_df = heat_df.dropna(axis=0, subset=['Y','X'])

# List comprehension to make out list of lists
heat_data = [[row['Lat_rand'],row['Lon_rand']] for index, row in heat_df.iterrows()]

# Create weight column, using date
heat_df['Weight'] = df3['Hour']
heat_df['Weight'] = heat_df['Weight'].astype(float)
heat_df = heat_df.dropna(axis=0, subset=['Lat_rand','Lon_rand', 'Weight'])

# List comprehension to make out list of lists
heat_data = [[[row['Lat_rand'],row['Lon_rand']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(0,24)]

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(map_hooray)
# Display the map
map_hooray


# # Using data visualizations to check for data quality issues 
# 
# # (spotting bad data)

# ## Yearly distribution per Road
# * The main feature that is clearly visible is the drop in taffic on nearly all roads on July, which is the month where most Copenhageners are on holidays away from the city
# * a few roads that except these rule show unclear patterns that could be due to data quality issues (check if this is true after wi enclose other years

# In[ ]:


y = df.groupby(["Year", "Vejnavn"]).sum()["Total Vehicles"].unstack()
#y


# In[ ]:


y.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# ## Montly distribution per Road (Net Entry-Exit Flows)

# In[ ]:


md = df.groupby(["Month", "Vejnavn"]).sum()["Net Traffic Flow"].unstack()
md


# In[ ]:


md.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# ## Weekly distribution per Road (Entry-Exit flows)

# In[ ]:


wd = df.groupby(["WeekDay", "Vejnavn"]).sum()["Net Traffic Flow"].unstack()
wd


# In[ ]:


wd.plot(kind='bar', subplots=True, figsize=(15,60), layout=(9,4))


# ### adding GeoJson file: example
# https://medium.com/tech-carnot/interactive-map-based-visualization-using-plotly-44e8ad419b97

# In[ ]:


import plotly.express as px
import geopandas as gpd
import json
import ogr
import pandas as pd


# ### esempio india
# 
# https://medium.com/tech-carnot/interactive-map-based-visualization-using-plotly-44e8ad419b97
# 

# In[ ]:


url = 'https://raw.githubusercontent.com/carnot-technologies/MapVisualizations/master/json_files/India_States_2020_compressed.json'

import urllib.request, json 
import pandas as pd

#url = ''
with urllib.request.urlopen("https://raw.githubusercontent.com/carnot-technologies/MapVisualizations/master/json_files/India_States_2020_compressed.json") as url:
    India_states  = json.loads(url.read().decode())
#    df = pd.DataFrame(data)
#print(df)
print(India_states)


# In[ ]:


#Have a look at the features
India_states["features"][0].keys()


# In[ ]:


#Let us look at just one location:
India_states["features"][0]['geometry']


# In[ ]:





# ### esempio con dati CPH

# In[ ]:


# GoeJson file downloaded from here: https://giedriusk.carto.com/tables/copenhagen_districts/public

#with open('copenhagen_districts.geojson') as f:
#    cph_districts = json.load(f)
    
#cph_districts


# In[ ]:


# iport geojson files example
# https://stackoverflow.com/questions/59306252/importing-json-file-url-to-pandas-data-frame
# geojsone file uploaded on Lars' GitHub page: https://github.com/LarsBryld/socialdata

import urllib.request, json 

with urllib.request.urlopen("https://raw.githubusercontent.com/LarsBryld/socialdata/main/copenhagen_districts.geojson") as url:
    cph_districts = json.loads(url.read().decode())

#print(cph_districts)


# In[ ]:


#Have a look at the features
cph_districts["features"][0].keys()


# In[ ]:


#Let us look at just one location:
cph_districts["features"][0]['geometry']


# In[ ]:


cph_districts["features"][0]['properties']['name']


# In[ ]:


cph_districts["features"][0]['geometry']['coordinates']#[0]#[0]#[0]


# In[ ]:


len(cph_districts["features"])


# In[ ]:


districts = []
for i in range(len(cph_districts["features"])):
    districts.append(cph_districts["features"][i]['properties']['name'])
    
districts


# In[ ]:


# CPH GeoJson file downloaded from here: 
# https://giedriusk.carto.com/tables/copenhagen_districts/public

# checking which roads belongs to which districts using this:


# In[ ]:


# if the mapping is successful, build the sample df needed for Plotly Maps otherwise, create a make up df for the purpose

# build the static map using:
# https://medium.com/tech-carnot/interactive-map-based-visualization-using-plotly-44e8ad419b97

# build the slider map mixing the above and below:
# https://amaral.northwestern.edu/blog/step-step-how-plot-map-slider-represent-time-evolu

# hopefully everythig will be all right :D


# In[ ]:


dfd = pd.DataFrame(districts)
#df.rename(columns={"A": "a", "B": "c"})
dfd = dfd.rename(columns={0: "Districts"})
for i in range(5,15):
    dfd['20' + str(i)] = np.random.uniform(1000000,2000000,13)
    
dfd = dfd.rename(columns={'205': '2005','206': '2006','207': '2007','208': '2008','209': '2009'})
dfd


# In[ ]:


import plotly.express as px

max_value = dfd['2005'].max()
fig = px.choropleth(dfd, locations='Districts',
                    geojson=cph_districts, featureidkey="properties.name",
                           color='2005',
                           color_continuous_scale="Viridis",
                           range_color=(0, max_value),
                           projection="mercator"
                          )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### let's now create a slider for interactive purposes

# In[ ]:


# fisrt we need to turn the colunms data into rows data (by year)

dfd = dfd.melt(id_vars=["Districts"],
        value_vars=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014'],
        var_name="Year", 
        value_name="Vehicles")

# convert year coulns from string to int
dfd['Year'] = pd.to_numeric(dfd['Year'])

dfd


# In[ ]:


#import all packages

import pandas as pd
import plotly
import plotly.graph_objs as go

import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)


# In[ ]:


# first we develop another version of the single Year

year = 2005

dfd_sected = dfd[(dfd['Year']== year )]
dfd_sected


# In[ ]:


# selecting the Choropleth Maps + hoover text

scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'], 
       [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds

for col in dfd_sected.columns:
    dfd_sected[col] = dfd_sected[col].astype(str)

dfd_sected['text'] = dfd_sected['Districts'] + ' total Vehicles: ' + dfd_sected['Vehicles']


# In[ ]:


# creating the dictionary needed for the Choropleth Maps

data = [ dict(
            type='choropleth', # type of map-plot
            colorscale = scl,
            autocolorscale = False,
            locations = dfd_sected['Districts'], # the column with the state
            z = dfd_sected['Vehicles'].astype(float), # the variable I want to color-code
            locationmode = 'ISO-3',
            text = dfd_sected['text'], # hover text
            marker = dict(     # for the lines separating states
                        line = dict (
                                  color = 'rgb(255,255,255)', 
                                  width = 2) ),               
            colorbar = dict(
                        title = "Total Vehicles")
            ) 
       ]


# In[ ]:


# layout and plotting

layout = dict(
        title = year,
        geo = dict(margin={"r":0,"t":0,"l":0,"b":0}
#            scope='usa',
#            projection=dict( type='albers usa' ),

#              showlakes = True,  # if you want to give color to the lakes

#             lakecolor = 'rgb(73, 216, 230)'  
            ),
             )

fig = dict( data=data, layout=layout )

plotly.offline.iplot(fig)


# In[ ]:


# layout and plotting

layout = dict(
        title = year,
#        geo = dict(
#    showcountries=True, countrycolor="Black",
#    showsubunits=True, subunitcolor="Blue"
##            scope='usa',
#            projection=dict( type='albers usa' ),

#              showlakes = True,  # if you want to give color to the lakes

#             lakecolor = 'rgb(73, 216, 230)'  
#            ),#fitbounds="locations", visible=False
             )

fig = dict( data=data, layout=layout )

plotly.offline.iplot(fig)


# In[ ]:


# layout and plotting

layout = dict(
        title = year,
#        geo = dict(
#            scope='usa',
#            projection=dict( type='albers usa' ),

#              showlakes = True,  # if you want to give color to the lakes

#             lakecolor = 'rgb(73, 216, 230)'  
#            ),fitbounds="locations", visible=False
             )

fig = dict( data=data, layout=layout )

plotly.offline.iplot(fig)


# In[ ]:


data


# https://plotly.com/python/reference/layout/
# 
# https://plotly.com/python/reference/layout/geo/

# In[ ]:





# In[ ]:





# In[ ]:


import plotly.express as px

max_value = dfd['2005'].max()
fig = px.choropleth(dfd, locations='Districts',
                    geojson=cph_districts, featureidkey="properties.name",
                           color='2005',
                           color_continuous_scale="Viridis",
                           range_color=(0, max_value),
                           projection="mercator"
                          )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


import plotly.express as px

df = px.data.election()
geojson = px.data.election_geojson()

fig = px.choropleth(df, geojson=geojson, color="Bergeron",
                    locations="district", featureidkey="properties.district",
                    projection="mercator"
                   )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

