# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

data = pd.read_csv(path)
data['Rating'].hist()

data = data[data['Rating'] <= 5]
data['Rating'].hist()

#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())*100
missing_data = pd.concat([total_null, percent_null], keys = ['Total', 'Percent'], axis = 1)
print(missing_data)

data.dropna(axis = 0 , inplace = True)

total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())*100
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys = ['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
plt.figure(figsize = (20,20))
sns.catplot(x = 'Category', y = 'Rating', data = data)
sns.set(style="ticks")
plt.xticks(rotation = 90, fontsize = 9)
plt.title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data['Installs'].value_counts()
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].str.replace(',', "")
data['Installs'] = data['Installs'].astype(int)

le = LabelEncoder()
le.fit(data['Installs'])
data['Installs'] = le.transform(data['Installs'])

plt.figure(figsize = (14,10))
a = sns.regplot(x = "Installs", y = 'Rating', data = data)
a.axes.set_title('Rating vs Installs [RegPlot]', fontsize = 20)
a.set_xlabel("Installs", fontsize = 18) 
a.set_ylabel("Rating", fontsize = 18)
a.tick_params(labelsize = 8)
#Code ends here



# --------------
#Code starts here
data['Price'].value_counts()
data['Price'] = data['Price'].str.replace("$","")

data['Price'] = data['Price'].astype(float)

plt.figure(figsize = (14,10))
a = sns.regplot(x = "Price", y = 'Rating', data = data)
a.axes.set_title('Rating vs Price [RegPlot]', fontsize = 20)
a.set_xlabel("Price", fontsize = 18) 
a.set_ylabel("Rating", fontsize = 18)
a.tick_params(labelsize = 8)
#Code ends here


# --------------

#Code starts here
unique_genre = data['Genres'].unique()
print(unique_genre)

a = []
for i in data['Genres']:
    a.append(i.split(";")[0])

data['Genres'] = a

gr_mean = data.groupby('Genres', as_index = False)['Rating'].mean()
gr_mean.describe()

gr_mean = gr_mean.sort_values(by = 'Rating')
print(gr_mean.iloc[0], gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here

data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()

LastUpdatedDays = (max_date - data['Last Updated']).dt.days
data["Last Updated Days"] = LastUpdatedDays

plt.figure(figsize = (14,10))
a = sns.regplot(x = "Last Updated Days", y = 'Rating', data = data)
a.set_xlabel('Last Updated Days', fontsize = 15)
a.set_ylabel('Rating', fontsize = 15)
a.axes.set_title('Rating vs Last Updated [RegPlot]')
a.tick_params(labelsize = 13)
#Code ends here


