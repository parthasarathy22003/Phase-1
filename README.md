# Phase-1
Contains phase 1 file

ADS_PHASE_1
https://drive.google.com/file/d/1rWvtJuTB6BNSpFd3QuPwOPrUol_FhL46/view?usp=drivesdk

ADS_PHASE_2
https://drive.google.com/file/d/1rZcInqzNIejqyUwzx6ynV8VLDBRA7VOA/view?usp=drivesdk

ADS_PHASE_3
https://docs.google.com/document/d/1rifMAEurKByyfTQQ84m2f2sjIoJZsPwA/edit?usp=drivesdk&ouid=113631827061482522051&rtpof=true&sd=true

ADS_PHASE_4
https://docs.google.com/document/d/1rvxbdz6Gmfy9kAUkZGPPL-iQKipoJtYC/edit?usp=drivesdk&ouid=113631827061482522051&rtpof=true&sd=true

ADS_PHASE_5
https://docs.google.com/document/d/1rje-2eUe2pqcG-5e_7M79o7e-TDmxzEZ/edit?usp=drivesdk&ouid=113631827061482522051&rtpof=true&sd=true


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import requests
import warnings
warnings.filterwarnings('ignore')
read_data = requests.get(url).content
[2]
address = pd.read_csv(io.StringIO(read_data.decode('utf-8')))
address.head()

[4]
vaccine=pd.read_csv(io.StringIO(read_data.decode('utf-8')))
[5]
data=address
[6]
data.columns
Index(['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
       'new_cases_smoothed', 'total_deaths', 'new_deaths',
       'new_deaths_smoothed', 'total_cases_per_million',
       'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million',
       'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations',
       'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
       'new_vaccinations', 'new_vaccinations_smoothed',
       'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
       'new_vaccinations_smoothed_per_million',
       'new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred', 'stringency_index',
       'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million'],
      dtype='object')
[7]
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 191376 entries, 0 to 191375
Data columns (total 67 columns):
 #   Column                                      Non-Null Count   Dtype  
---  ------                                      --------------   -----  
 0   iso_code                                    191376 non-null  object 
 1   continent                                   180250 non-null  object 
 2   location                                    191376 non-null  object 
 3   date                                        191376 non-null  object 
 4   total_cases                                 183834 non-null  float64
 5   new_cases                                   183621 non-null  float64
 6   new_cases_smoothed                          182447 non-null  float64
 7   total_deaths                                165368 non-null  float64
 8   new_deaths                                  165361 non-null  float64
 9   new_deaths_smoothed                         164198 non-null  float64
 10  total_cases_per_million                     182986 non-null  float64
 11  new_cases_per_million                       182773 non-null  float64
 12  new_cases_smoothed_per_million              181604 non-null  float64
 13  total_deaths_per_million                    164533 non-null  float64
 14  new_deaths_per_million                      164526 non-null  float64
 15  new_deaths_smoothed_per_million             163368 non-null  float64
 16  reproduction_rate                           140710 non-null  float64
 17  icu_patients                                25496 non-null   float64
 18  icu_patients_per_million                    25496 non-null   float64
 19  hosp_patients                               26747 non-null   float64
 20  hosp_patients_per_million                   26747 non-null   float64
 21  weekly_icu_admissions                       6222 non-null    float64
 22  weekly_icu_admissions_per_million           6222 non-null    float64
 23  weekly_hosp_admissions                      12397 non-null   float64
 24  weekly_hosp_admissions_per_million          12397 non-null   float64
 25  total_tests                                 77683 non-null   float64
 26  new_tests                                   74008 non-null   float64
 27  total_tests_per_thousand                    77683 non-null   float64
 28  new_tests_per_thousand                      74008 non-null   float64
 29  new_tests_smoothed                          101315 non-null  float64
 30  new_tests_smoothed_per_thousand             101315 non-null  float64
 31  positive_rate                               93441 non-null   float64
 32  tests_per_case                              91681 non-null   float64
 33  tests_units                                 104079 non-null  object 
 34  total_vaccinations                          52388 non-null   float64
 35  people_vaccinated                           49909 non-null   float64
 36  people_fully_vaccinated                     47375 non-null   float64
 37  total_boosters                              24452 non-null   float64
 38  new_vaccinations                            42912 non-null   float64
 39  new_vaccinations_smoothed                   103578 non-null  float64
 40  total_vaccinations_per_hundred              52388 non-null   float64
 41  people_vaccinated_per_hundred               49909 non-null   float64
 42  people_fully_vaccinated_per_hundred         47375 non-null   float64
 43  total_boosters_per_hundred                  24452 non-null   float64
 44  new_vaccinations_smoothed_per_million       103578 non-null  float64
 45  new_people_vaccinated_smoothed              102491 non-null  float64
 46  new_people_vaccinated_smoothed_per_hundred  102491 non-null  float64
 47  stringency_index                            148621 non-null  float64
 48  population                                  190211 non-null  float64
 49  population_density                          170524 non-null  float64
 50  median_age                                  158052 non-null  float64
 51  aged_65_older                               156377 non-null  float64
 52  aged_70_older                               157223 non-null  float64
 53  gdp_per_capita                              157205 non-null  float64
 54  extreme_poverty                             102625 non-null  float64
 55  cardiovasc_death_rate                       157692 non-null  float64
 56  diabetes_prevalence                         165401 non-null  float64
 57  female_smokers                              119268 non-null  float64
 58  male_smokers                                117633 non-null  float64
 59  handwashing_facilities                      77477 non-null   float64
 60  hospital_beds_per_thousand                  139914 non-null  float64
 61  life_expectancy                             178964 non-null  float64
 62  human_development_index                     153621 non-null  float64
 63  excess_mortality_cumulative_absolute        6553 non-null    float64
 64  excess_mortality_cumulative                 6553 non-null    float64
 65  excess_mortality                            6553 non-null    float64
 66  excess_mortality_cumulative_per_million     6553 non-null    float64
dtypes: float64(62), object(5)
memory usage: 97.8+ MB

[8]
data.describe(include='all')

[9]
vaccine.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 42395 entries, 0 to 42394
Data columns (total 4 columns):
 #   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   location            42395 non-null  object
 1   date                42395 non-null  object
 2   vaccine             42395 non-null  object
 3   total_vaccinations  42395 non-null  int64 
dtypes: int64(1), object(3)
memory usage: 1.3+ MB

[10]
vaccine.describe()

2.data preprocessing
[11]
data.isnull().sum()
iso_code                                        0
continent                                   11126
location                                        0
date                                            0
total_cases                                  7542
                                            ...  
human_development_index                     37755
excess_mortality_cumulative_absolute       184823
excess_mortality_cumulative                184823
excess_mortality                           184823
excess_mortality_cumulative_per_million    184823
Length: 67, dtype: int64
[12]
data['date']=pd.to_datetime(data['date'])
[13]
vaccine['date']=pd.to_datetime(data['date'])
[14]
data.drop([ 'new_cases_smoothed','new_deaths_smoothed', 'new_cases_smoothed_per_million',
       'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'new_vaccinations_smoothed',
       'new_vaccinations_smoothed_per_million',
       'new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred'], axis=1, inplace=True)
[15]
data.drop(['icu_patients_per_million','hosp_patients','hosp_patients_per_million','weekly_icu_admissions',
           'weekly_icu_admissions_per_million','weekly_hosp_admissions','weekly_hosp_admissions_per_million',
          'new_tests_per_thousand','excess_mortality_cumulative_absolute','excess_mortality_cumulative',
         'excess_mortality','excess_mortality_cumulative_per_million','stringency_index','life_expectancy','human_development_index','extreme_poverty',                        
'cardiovasc_death_rate',                  
'diabetes_prevalence',                 
'female_smokers',                         
'male_smokers', 
'handwashing_facilities', 
'hospital_beds_per_thousand'],axis= 1,inplace=True)
checking for the null values
[16]
x=data.isnull().sum()*100/len(data)
x
iso_code                                0.000000
continent                               5.813686
location                                0.000000
date                                    0.000000
total_cases                             3.940933
new_cases                               4.052232
total_deaths                           13.590001
new_deaths                             13.593659
total_cases_per_million                 4.384040
new_cases_per_million                   4.495339
total_deaths_per_million               14.026315
new_deaths_per_million                 14.029972
total_tests                            59.408181
new_tests                              61.328484
total_tests_per_thousand               59.408181
positive_rate                          51.174128
tests_per_case                         52.093784
tests_units                            45.615438
total_vaccinations                     72.625617
people_vaccinated                      73.920972
people_fully_vaccinated                75.245067
total_boosters                         87.223058
new_vaccinations                       77.577126
total_vaccinations_per_hundred         72.625617
people_vaccinated_per_hundred          73.920972
people_fully_vaccinated_per_hundred    75.245067
total_boosters_per_hundred             87.223058
population                              0.608749
population_density                     10.895828
median_age                             17.412842
aged_65_older                          18.288082
aged_70_older                          17.846020
gdp_per_capita                         17.855426
dtype: float64
checking for duplicate values
[17]
duplicate = data[data.duplicated()] 
duplicate

[18]
print(data.isnull().values.any()) 
True

[19]

data['total_deaths'].mean()
64774.858037830774
[20]
data['total_deaths'].median()
917.0
[21]
data['total_deaths'].replace(np.nan,data['total_deaths'].median()).head(10)
0    917.0
1    917.0
2    917.0
3    917.0
4    917.0
5    917.0
6    917.0
7    917.0
8    917.0
9    917.0
Name: total_deaths, dtype: float64
using bfill method to fill nan cells
[22]
data.fillna(method="bfill")


[23]
data.isnull().values.any() #Checking fo nan values in whole dataframe

True
[24]

data.head()

[25]
data.info(
)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 191376 entries, 0 to 191375
Data columns (total 33 columns):
 #   Column                               Non-Null Count   Dtype         
---  ------                               --------------   -----         
 0   iso_code                             191376 non-null  object        
 1   continent                            180250 non-null  object        
 2   location                             191376 non-null  object        
 3   date                                 191376 non-null  datetime64[ns]
 4   total_cases                          183834 non-null  float64       
 5   new_cases                            183621 non-null  float64       
 6   total_deaths                         165368 non-null  float64       
 7   new_deaths                           165361 non-null  float64       
 8   total_cases_per_million              182986 non-null  float64       
 9   new_cases_per_million                182773 non-null  float64       
 10  total_deaths_per_million             164533 non-null  float64       
 11  new_deaths_per_million               164526 non-null  float64       
 12  total_tests                          77683 non-null   float64       
 13  new_tests                            74008 non-null   float64       
 14  total_tests_per_thousand             77683 non-null   float64       
 15  positive_rate                        93441 non-null   float64       
 16  tests_per_case                       91681 non-null   float64       
 17  tests_units                          104079 non-null  object        
 18  total_vaccinations                   52388 non-null   float64       
 19  people_vaccinated                    49909 non-null   float64       
 20  people_fully_vaccinated              47375 non-null   float64       
 21  total_boosters                       24452 non-null   float64       
 22  new_vaccinations                     42912 non-null   float64       
 23  total_vaccinations_per_hundred       52388 non-null   float64       
 24  people_vaccinated_per_hundred        49909 non-null   float64       
 25  people_fully_vaccinated_per_hundred  47375 non-null   float64       
 26  total_boosters_per_hundred           24452 non-null   float64       
 27  population                           190211 non-null  float64       
 28  population_density                   170524 non-null  float64       
 29  median_age                           158052 non-null  float64       
 30  aged_65_older                        156377 non-null  float64       
 31  aged_70_older                        157223 non-null  float64       
 32  gdp_per_capita                       157205 non-null  float64       
dtypes: datetime64[ns](1), float64(28), object(4)
memory usage: 48.2+ MB

[26]
data.drop(['tests_units'],axis=1,inplace=True)
[27]
null_percentage=data.isna().sum()*100/len(data)
null_percentage.head(38)
iso_code                                0.000000
continent                               5.813686
location                                0.000000
date                                    0.000000
total_cases                             3.940933
new_cases                               4.052232
total_deaths                           13.590001
new_deaths                             13.593659
total_cases_per_million                 4.384040
new_cases_per_million                   4.495339
total_deaths_per_million               14.026315
new_deaths_per_million                 14.029972
total_tests                            59.408181
new_tests                              61.328484
total_tests_per_thousand               59.408181
positive_rate                          51.174128
tests_per_case                         52.093784
total_vaccinations                     72.625617
people_vaccinated                      73.920972
people_fully_vaccinated                75.245067
total_boosters                         87.223058
new_vaccinations                       77.577126
total_vaccinations_per_hundred         72.625617
people_vaccinated_per_hundred          73.920972
people_fully_vaccinated_per_hundred    75.245067
total_boosters_per_hundred             87.223058
population                              0.608749
population_density                     10.895828
median_age                             17.412842
aged_65_older                          18.288082
aged_70_older                          17.846020
gdp_per_capita                         17.855426
dtype: float64
[28]
data=data.fillna(method="bfill")
[29]
null_percentage=data.isna().sum()*100/len(data)

null_percentage.head(38)
iso_code                               0.000000
continent                              0.000000
location                               0.000000
date                                   0.000000
total_cases                            0.000000
new_cases                              0.000000
total_deaths                           0.000000
new_deaths                             0.000000
total_cases_per_million                0.000000
new_cases_per_million                  0.000000
total_deaths_per_million               0.000000
new_deaths_per_million                 0.000000
total_tests                            0.000523
new_tests                              0.007838
total_tests_per_thousand               0.000523
positive_rate                          0.000523
tests_per_case                         0.000523
total_vaccinations                     0.001045
people_vaccinated                      0.001045
people_fully_vaccinated                0.001045
total_boosters                         0.001045
new_vaccinations                       0.001045
total_vaccinations_per_hundred         0.001045
people_vaccinated_per_hundred          0.001045
people_fully_vaccinated_per_hundred    0.001045
total_boosters_per_hundred             0.001045
population                             0.000000
population_density                     0.000000
median_age                             0.000000
aged_65_older                          0.000000
aged_70_older                          0.000000
gdp_per_capita                         0.000000
dtype: float64
[30]
data.isnull().sum()
iso_code                                0
continent                               0
location                                0
date                                    0
total_cases                             0
new_cases                               0
total_deaths                            0
new_deaths                              0
total_cases_per_million                 0
new_cases_per_million                   0
total_deaths_per_million                0
new_deaths_per_million                  0
total_tests                             1
new_tests                              15
total_tests_per_thousand                1
positive_rate                           1
tests_per_case                          1
total_vaccinations                      2
people_vaccinated                       2
people_fully_vaccinated                 2
total_boosters                          2
new_vaccinations                        2
total_vaccinations_per_hundred          2
people_vaccinated_per_hundred           2
people_fully_vaccinated_per_hundred     2
total_boosters_per_hundred              2
population                              0
population_density                      0
median_age                              0
aged_65_older                           0
aged_70_older                           0
gdp_per_capita                          0
dtype: int64
[31]
data['new_tests'].replace(np.nan,data['new_tests'].median(),inplace=True)
data['positive_rate'].replace(np.nan,data['positive_rate'].median(),inplace=True)
data['tests_per_case'].replace(np.nan,data['tests_per_case'].median(),inplace=True)
data['new_vaccinations'].replace(np.nan,data['new_vaccinations'].median(),inplace=True)

[32]
data.isnull().sum()
iso_code                               0
continent                              0
location                               0
date                                   0
total_cases                            0
new_cases                              0
total_deaths                           0
new_deaths                             0
total_cases_per_million                0
new_cases_per_million                  0
total_deaths_per_million               0
new_deaths_per_million                 0
total_tests                            1
new_tests                              0
total_tests_per_thousand               1
positive_rate                          0
tests_per_case                         0
total_vaccinations                     2
people_vaccinated                      2
people_fully_vaccinated                2
total_boosters                         2
new_vaccinations                       0
total_vaccinations_per_hundred         2
people_vaccinated_per_hundred          2
people_fully_vaccinated_per_hundred    2
total_boosters_per_hundred             2
population                             0
population_density                     0
median_age                             0
aged_65_older                          0
aged_70_older                          0
gdp_per_capita                         0
dtype: int64
[33]
v=vaccine.drop(['total_vaccinations'], axis = 1)
v

Integrating two datasets,vaccines and vaccination names
[34]
final=pd.merge(data,v,on=['date','location'],how='left')
[35]
final.isnull().sum()
iso_code                                    0
continent                                   0
location                                    0
date                                        0
total_cases                                 0
new_cases                                   0
total_deaths                                0
new_deaths                                  0
total_cases_per_million                     0
new_cases_per_million                       0
total_deaths_per_million                    0
new_deaths_per_million                      0
total_tests                                 1
new_tests                                   0
total_tests_per_thousand                    1
positive_rate                               0
tests_per_case                              0
total_vaccinations                          2
people_vaccinated                           2
people_fully_vaccinated                     2
total_boosters                              2
new_vaccinations                            0
total_vaccinations_per_hundred              2
people_vaccinated_per_hundred               2
people_fully_vaccinated_per_hundred         2
total_boosters_per_hundred                  2
population                                  0
population_density                          0
median_age                                  0
aged_65_older                               0
aged_70_older                               0
gdp_per_capita                              0
vaccine                                168644
dtype: int64
[36]
n2=final.isna().sum()*100/len(final)
[37]
final['vaccine'].isna().sum()/len(final)*100

79.99165192314078
[38]
fin=pd.merge(data,v,on=['date','location'],how='left')
fin

one hot encoding
[39]
# Own implementation of One Hot Encoding - Data Transformation
def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:5] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1

    return df

# One Hot Encoding
print("One Hot Encoding categorical data...")
columns_to_convert = ['vaccine']

for column in columns_to_convert:
    df_all = convert_to_binary(df=final, column_to_convert=column)
    df_all.drop(column, axis=1, inplace=True)
print("One Hot Encoding categorical data...completed")
One Hot Encoding categorical data...
One Hot Encoding categorical data...completed

[93]
fin=fin.dropna()
fin

[41]
idf2=vaccine.groupby('vaccine',as_index=False).sum()
[42]
idf2=idf2[['vaccine','total_vaccinations']]

[43]
idf2.total_vaccinations[0]
165653252
[44]
idf2

[45]
idf=vaccine['vaccine']
[46]
idf=idf.to_frame()
[47]
idf=idf.dropna()
[48]
idf

3. Questions
1.which is the most used vaccines?
[49]
sns.set_theme(style="darkgrid")
sns.set(rc = {'figure.figsize':(20,10)})
ax = sns.countplot(x="vaccine", data=vaccine)


pfizer is the most used vaccine because it got approved in many countries very quickly.







2. Which countries has highest people fully vaccinated per hundred
[50]
c=final['location'].value_counts().loc[lambda x:x>1500]
c=pd.DataFrame(c)
c.rename(columns={'location':"people_fully_vaccinated_per_hundred"},inplace=True)
c[1:]

[51]
plt.style.use("fivethirtyeight")
plt.figure(figsize=(15,8))
plt.xlabel("Country")
plt.ylabel("people_fully_vaccinated_per_hundred")
sns.barplot(y=c['people_fully_vaccinated_per_hundred'][1:],x=c.index[1:])
plt.show()






3. what is the share of total vaccinationsof covid-19 in each country
[52]
df_loc=final.groupby('location',as_index=False)
[53]
fig = px.treemap(final, path=[px.Constant('total_vaccinations'),'location'], values='total_vaccinations',
                   hover_data=['location'])
fig.show()
8 countries in top 10 countries with people_fully_vaccinatd_per_hundred belong to europe








5. Data Analysis and Visualisation
[54]
corrmat = final.corr()
  
f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
<AxesSubplot:>

Correlation describes how two attributes are related and indicates that as one variable changes in value, the other variable tends to change in a specific direction
[55]
time_series = pd.DataFrame(final['date'].value_counts().reset_index())
time_series.columns = ['date', 'count']
[56]
time_series= time_series.sort_values('date', ascending=True)
plt.style.use("fivethirtyeight")
plt.figure(figsize=(15,8))
plt.plot(time_series['date'], time_series['count'],linewidth=1)
plt.xticks(rotation='vertical')
plt.xlabel("Date")
plt.ylabel("Count")
Text(0, 0.5, 'Count')

[57]
a=final['median_age'].values
[58]
d=final['total_boosters'].values
[59]
X=final[['date','total_vaccinations_per_hundred']]
[60]
X

[61]
df=X
[62]
df

[63]
final.plot(x='date',y='total_boosters')
<AxesSubplot:xlabel='date'>

Booster doses drive started around november 2021.
[64]
final[['date','total_vaccinations_per_hundred']]

[65]
df['total_vaccinations']=final['total_vaccinations']
[66]
df[['total_vaccinations','total_vaccinations_per_hundred']]=final[['total_vaccinations','total_vaccinations_per_hundred']]
[67]
df

[68]
temp=pd.DataFrame()
[69]
temp[['date','total_vaccinations','total_vaccinations_per_hundred']]=final[['date','total_vaccinations','total_vaccinations_per_hundred']]
[70]
temp

[71]
final.plot(x='date',y='total_vaccinations')
<AxesSubplot:xlabel='date'>

there is a linear increase of total vaccinations
[72]

# temp[['date','total_vaccinations']].plot(kind='kde')

[73]
df

[74]

temp=temp.set_index('date')
[75]
temp


[76]
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
[77]
df2=temp
[78]
df2

[79]
df_loc=final[['date' ]]
[80]
df_loc=final.groupby('location',as_index=False).sum()
[81]
df_loc

[82]
fig = px.treemap(df_loc, path=[px.Constant('gdp_per_capita'),'location'], values='gdp_per_capita',
                   hover_data=['location'])
fig.show()
Germany,Switzerland,Qatar,Luxembourg has Highest gdp per capita
df_loc.columns

[83]
fig = px.treemap(df_loc, path=[px.Constant('total_deaths'),'location'], values='total_deaths',
                   hover_data=['location'])
fig.show()
European union has the highest deaths , even though it has the higher vaccinations and Gdp
[84]
df_add=address.groupby('iso_code',as_index=False).sum()
[85]
df_add

[86]
map_total_vac = px.choropleth(data_frame = df_add , locations="iso_code" , color="total_vaccinations_per_hundred" 
                             , hover_name="iso_code" , color_continuous_scale=px.colors.sequential.deep)
map_total_vac.update_layout(title_text='Total vaccinations per hundred in each country'
                                  , title_font={'family':'serif','size':26} , title = {'y':0.94 , 'x':0.45})
map_total_vac.show()


North America,European Union,China and countries with higher gdp have higher total vaccination per hundred
[87]
sns.set_theme(style="darkgrid")
sns.set(rc = {'figure.figsize':(40,20)})
ax = sns.countplot(x="location", data=vaccine)


european union has recorded highest number of vaccinations
[88]
df_add.columns
Index(['iso_code', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
       'total_cases_per_million', 'new_cases_per_million',
       'total_deaths_per_million', 'new_deaths_per_million', 'total_tests',
       'new_tests', 'total_tests_per_thousand', 'positive_rate',
       'tests_per_case', 'total_vaccinations', 'people_vaccinated',
       'people_fully_vaccinated', 'total_boosters', 'new_vaccinations',
       'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
       'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita'],
      dtype='object')
[89]
df

[90]
df_loc

[91]
fig = px.bar(df_loc.sort_values('new_deaths', ascending=False)[:20][::-1], 
             x='new_deaths', y='location',
             title=' New Deaths Worldwide', text='location', height=1000, orientation='h')
fig.show()

European Union has the highest new deaths
[94]

import plotly.express as px
fig = px.treemap(fin,names = 'location',values = 'total_vaccinations',
                 path = ['vaccine','location'],
                 title="Total Vaccinations per country grouped by Vaccines",
                 color_discrete_sequence =px.colors.qualitative.Set1)
fig.show()
Vaccines like pfizer,Moderna are used by many countries where as vaccines like sinovac and sputnik are not approved in many countries
[112]
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
df_con=final.groupby('continent',as_index=False).sum()
import matplotlib as mpl
mpl.rcParams['font.size'] = 30.0
explode = (0, 0.1, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(df_con['total_vaccinations_per_hundred'], labels=df_con['continent'], autopct='%1.1f%%',
        shadow=True, startangle=90,explode=explode,textprops={'fontsize': 35})


ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("total vaccinations per 100(continent wise)",pad=80,fontdict={'fontsize':35})
plt.show()

asia and europe have around 50 percent of all vaccinations
[95]
plt.figure(figsize=(20,7))
sns.lineplot(x="date",y="new_vaccinations",data=final)
plt.title("New Vaccines")
plt.show()

trend of new vaccinations in world




5. Model Building
!pip install pycaret
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()


from datetime import date , datetime , timedelta

import pycaret.regression as caret


import warnings
warnings.filterwarnings('ignore')
import io
import requests
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
read_data = requests.get(url).content
data_detailed = pd.read_csv(io.StringIO(read_data.decode('utf-8')))

url2 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations-by-manufacturer.csv"
read_data2 = requests.get(url2).content
data_total = pd.read_csv(io.StringIO(read_data2.decode('utf-8')))

print("* "*10+" data_detailed "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_detailed.shape[0] , data_detailed.shape[1]))
print(data_detailed.info())
print("* "*10+" data_total "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_total.shape[0] , data_total.shape[1]))
print(data_total.info())
data_detailed.date.max()
# find the last date
last_date = data_detailed.sort_values(by = 'date' , ascending=False)['date'].iloc[0]
# its ''2022-05-21'
countries = data_total.location.unique()

data_detailed[(data_detailed.date == last_date)&(data_detailed.people_fully_vaccinated_per_hundred.isnull())]
data_detailed[(data_detailed.date == last_date)&(data_detailed.location == 'Germany')]
euro_vaccines = data_total[(data_total.location == 'European Union') &
                         (data_total.date == last_date)][['vaccine','total_vaccinations']]
euro_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)
euro_vaccines
pie_euro_vac = go.Figure(data = go.Pie(values = euro_vaccines.total_vaccinations, 
                          labels = euro_vaccines.vaccine, hole = 0.55))
pie_euro_vac.update_traces(textposition='outside', textinfo='percent+label')
pie_euro_vac.update_layout(annotations=[dict(text='Vaccines used by', x=0.5, y=0.55, font_size=16, showarrow=False),
                                       dict(text='European Union', x=0.5, y=0.45, font_size=16, showarrow=False)])
pie_euro_vac.show()
data_detailed[data_detailed.location == 'Germany']['date'].max() , data_total[data_total.location == 'Germany']['date'].max()
germany_vaccines=data_total[(data_total.location=='Germany')&(data_total.date=='2022-05-18')][['vaccine','total_vaccinations']]
germany_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)
df_germany_info = data_detailed[data_detailed.location == 'Germany']
fig_germany = make_subplots(rows = 4 , cols = 2
    , specs=[[{"type": "pie","rowspan": 2}, {"type": "scatter","rowspan": 2}]
           ,[None , None]
           ,[{"type": "scatter","colspan": 2,"rowspan": 2}, None]
           ,[None ,
             None]]
                            
    , subplot_titles=[
        '', 
        'temp',
        'temp' # i will change the titles a few lines later ...
    ])

fig_germany.add_trace(go.Pie(labels = germany_vaccines.vaccine , values = germany_vaccines.total_vaccinations
                                   , hole = 0.5 , pull = [0,0.1,0.1,0.1] , title = "Vaccines" , titleposition='middle center'
                                   , titlefont = {'family':'serif' , 'size':18}
                                   , textinfo = 'percent+label' , textposition = 'inside')
                     , row = 1 , col = 1)

fig_germany.add_trace(go.Scatter(x = df_germany_info['date']
                                , y = df_germany_info['daily_vaccinations']
                                , name = "Daily vaccinations")
                     , row = 1 , col = 2)

fig_germany.add_trace(go.Scatter(x = df_germany_info['date']
                                , y = df_germany_info['people_fully_vaccinated_per_hundred']
                                , name = "Fully vaccinated people percentage"
                                 # <br> for the next line in hover
                                , hovertemplate = "<b>%{x}</b><br>" +"Fully vaccinated people = %{y:.2f} %" +"<extra></extra>")
                     , row = 3 , col = 1)


fig_germany.layout.annotations[0].update(text="Number of daily vaccinations" , x=0.75
                                         , font = {'family':'serif','size':20})

fig_germany.layout.annotations[1].update(text="Fully vaccinated people percentage" , x=0.25 
                                         , font = {'family':'serif','size':20})

fig_germany.update_yaxes(range=[0, 100], row=3, col=1)
fig_germany.update_layout(width = 950,height=600, showlegend=True)
fig_germany.update_layout(title_text='Germany abstract informations'
                                  ,title_font={'family':'serif','size':26} , title = {'x':0.25 , 'y':0.95})

fig_germany.show()
data = pd.DataFrame()
data['Date'] = pd.to_datetime(df_germany_info['date'])
data['Target'] = df_germany_info['people_fully_vaccinated_per_hundred']
data.reset_index(drop = True , inplace = True)
data.Date.min() , data.Date.max() , len(data)
from datetime import date, datetime

d0 = date(2020 , 12 , 27)
d1 = date(2022 , 5 , 18)
delta = d1 - d0

days = delta.days + 1
print(days)
data=data.dropna()
data.isnull().sum()
data['Series'] = np.arange(1 , len(data)+1)

# Shift1 is the previous value(Target) for each row :
data['Shift1'] = data.Target.shift(1)

# mean of the Target during 10 previous days :
window_len = 10
window = data['Shift1'].rolling(window = window_len)
means = window.mean()
data['Window_mean'] = means


# This approach will make some Missing values (for example we dont have the previous value for the first row)
data.dropna(inplace = True)
data.reset_index(drop = True , inplace=True)

dates = data['Date'] # we will need this

data = data[['Series' , 'Window_mean' , 'Shift1' , 'Target']]

data
# 50% for train & 50% for test
train = data.iloc[:230,:] 
test = data.iloc[230:,:]

train.shape , test.shape
setup = caret.setup(data = train , test_data = test , target = 'Target' , fold_strategy = 'timeseries'
                 , remove_perfect_collinearity = False , numeric_features = ['Series' , 'Window_mean' , 'Shift1'] 
                     , fold = 5 , session_id = 51)
best = caret.compare_models(sort = 'MAE' , turbo = False)
best = caret.tune_model(best)
_ = caret.predict_model(best)
# generate predictions on the original dataset
predictions = caret.predict_model(best , data=data)

# add a date column in the dataset
predictions['Date'] = dates

# line plot
fig = px.line(predictions.rename(columns = {'Target':'Actual' }), x='Date', y=["Actual"])
fig.update_layout(annotations=[dict(text='Test set', x='2022-4-15', y=30, font_size=20, showarrow=False)])
# add a vertical rectangle for test-set separation

fig.add_vrect(x0 = dates.iloc[230], x1 = dates.iloc[-1], fillcolor="grey", opacity=0.25, line_width=1)
fig.show()
Future forecasting
As we used lag and window features, forecasting the future is a little harder.
For example we dont have the previous value for 2022-5-29 since we dont know the target value at 2022-5-28
So we will start from the first future time step and both we make predictions and also fill the lag features for next time steps. (maybe something like recursive functions)
future = pd.DataFrame(columns = ['Series' , 'Window_mean' , 'Shift1'])
future['Series'] = np.arange(300,450) # for the next 150 time steps
future['Window_mean'] = np.nan
future['Shift1'] = np.nan

# initialize the first row :
#------------------------------
future.iloc[0,2] = data['Target'].max()
sum = 0
for i in range(window_len):
    sum += data.iloc[len(data)-1-i,3]
    
future.iloc[0,1] = sum/window_len
future
for j in range(len(future)):
    current_row = j
    next_row = j+1
    
    # for the next_row we are going to fill :
    # 1. Shift1 --> use currnet_row predicted value
    # 2. Window_mean
    
    if current_row != len(future)-1 :
        # fill Shift1 for the next_row
        future.iloc[next_row,2] = caret.predict_model(best , future.iloc[[current_row]])['Label']
#         print(future.iloc[next_row,2]-future.iloc[current_row,2])
        
        
        # fill the Window_mean for the next_row
        if next_row < 9 :
            sum = 0
            num_rows_from_data = window_len - (next_row + 1)
            num_rows_from_future = window_len - num_rows_from_data

            for i in range(num_rows_from_data):
                sum += data.iloc[len(data)-1-i , 2]


            for i in range(num_rows_from_future):
                sum += future.iloc[next_row - i , 2]

            future.iloc[next_row , 1] = sum/window_len


        elif next_row >= 9:
            sum = 0
            for i in range(window_len):
                sum += future.iloc[next_row-i,2]
            future.iloc[next_row,1] = sum/window_len
As you see in the above cell, in each row of the future data frame we have the previous value in 'Shift1' column.
So with a reverse shift of this column, we have the current value for each row.
future['Predicted'] = future['Shift1'].shift(-1)

start = datetime.strptime("2022-05-19", "%Y-%m-%d")
date_generated = [start + timedelta(days=x) for x in range(0, 150)]
date_list = []
for date in date_generated:
    date_list.append(date.strftime("%Y-%m-%d"))
    
future['Date'] = date_list

future = future[['Date' , 'Predicted']]
future.dropna(inplace = True)
future
fig = go.Figure(data=go.Scatter(x=df_germany_info['date'], y = df_germany_info['people_fully_vaccinated_per_hundred']
                                ,mode='lines', line_color='red' , name = 'Until now'))
fig.add_trace(go.Scatter(x=future['Date'], y=future['Predicted'],mode='lines', line=dict(color="#0000ff"), name = 'Future'))



fig.show()

About 13th october 2022
100% of people in Germany will get fully vaccinated ?! Maybe




6. conclusion
European union is one of the best example for the region which has many deaths and also which produced high number of total vaccinations. Also the gdp of countries of this region has not much differed after pandemic.

Comparing the Root-mean-square error (rmse) and coefficient of discrimination(R2) values of models, Huber regression is selected with least RMSE of 0.1678 and R2 value of 0.9974 as the best model to predict the total vaccinations of particular region over the time period.

According to our model,Total Vaccinations of Germany might be completed by 2nd week of October.




7. Future Scope
The country governments can improve the vaccination facilities and availibility of vaccines to different locations by demand of vaccines and by refering most used vaccines across different countries. The synthesis of current research will be helpful to researchers analyzing historical trends in the COVID-19 pandemic and individuals interested in better understanding and advocating for underserved communities across the globe.





9. References
https://ourworldindata.org/covid-vaccinations

https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/

https://www.sciencedirect.com/science/article/pii/S2211379721006197

https://pycaret.gitbook.io/docs/

https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations

https://towardsdatascience.com/regression-in-the-face-of-messy-outliers-try-huber-regressor-3a54ddc12516

https://www.nature.com/articles/s41598-022-05915-3

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253925

https://machinelearningmastery.com/robust-regression-for-machine-learning



SUMMARY:

# COVID -19 VACCINATIONS ANALYSIS

## PROBLEM STATEMENT

- Forecasting of time taken for completing 100% total vaccinations of particular region over the time period.
- By this, vaccine manufacturing companies get to know the prior requirements of vaccine which helps to produce
  the vaccines in large scale and complete the vaccination drive with in calculated time.

## Contributers

[B Maniyarasu](https://github.com/Maniyarasu2508/Maniyarasu2508)

[T Parthasarthy](https://github.com/parthasarathy22003/Phase-1.git)

[K Karan](https://github.com/Karan0611/Karan)

[K Kalaiyarasan](https://github.com/KALAIYARASAN77/Kalaiyarasan777)

[G Dhayanithi](https://github.com/dhayanithiitgithub/DNSFILE)

## DATASETS Source: [https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress]

## Questions Analysed:-

### Which is the most used vaccines?

![image](https://user-images.githubusercontent.com/78417411/199703753-0a902b62-eca4-4495-9806-2fabe83c670d.png)

### Which countries has highest people fully vaccinated per hundred?

![image](https://user-images.githubusercontent.com/78417411/199704405-3bd55fda-6c4e-485a-adc7-775775ca863d.png)

### Corelation in dataset

![image](https://user-images.githubusercontent.com/78417411/199704615-a9f4c0c3-a32d-4524-98e5-3a2a3de12173.png)

### Correlation describes how two attributes are related and indicates that as one variable changes in value, the other variable tends to change in a specific direction

### Vaccination count

![image](https://user-images.githubusercontent.com/78417411/199704913-2ec7e213-0f31-41c5-a15b-2054cfbe2fa9.png)

### Booster dose count

![image](https://user-images.githubusercontent.com/78417411/199705190-c8373f8c-3e5f-4541-8622-d9e7547a671e.png)

### Total vaccination per hundred

![image](https://user-images.githubusercontent.com/78417411/199705500-b813d1bf-15bf-4a4e-a0c7-3fc8c0bd2787.png)

### Total vaccination per country grouped by vaccines

![image](https://user-images.githubusercontent.com/78417411/199705663-82a1e4f3-452b-46ec-a0ac-e5b5f66db23e.png)

### Total vaccination per hundred in each country

![image](https://user-images.githubusercontent.com/78417411/199705884-28fb9b3c-409c-45c7-aa82-df8ac02c147d.png)

### new vaccination rate

![image](https://user-images.githubusercontent.com/78417411/199706366-24273a27-47f2-4145-aa82-19b1edcc4b4a.png)

### Germany analysis

![image](https://user-images.githubusercontent.com/78417411/199706485-c31e63fd-9653-4869-82e1-dded85438015.png)

### Comparing with models

![image](https://user-images.githubusercontent.com/78417411/199706867-b9552974-f681-49bf-9800-34c50d093fca.png)

### Before forcasting

![image](https://user-images.githubusercontent.com/78417411/199707034-de16903a-474b-48fd-b8e6-edec7ad95b8a.png)

### After forecasting

![image](https://user-images.githubusercontent.com/78417411/199707135-efe5cbb3-a0c5-4f17-8779-6c316145ff69.png)
