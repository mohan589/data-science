import pandas as pd

df = pd.read_csv('nyc_weather.csv')
print(df.head())
print(f'Max Temparature { df['Temperature'].max() }')
print(f'Wind Speed {df['WindSpeedMPH'].mean()}')
df.fillna(0, inplace=True)
print(f'Wind Speed {df['WindSpeedMPH'].mean()}')

df = pd.read_csv('nyc_weather.csv',skiprows=2)
print(df.head())

# Remove headers

df = pd.read_csv('nyc_weather.csv',skiprows=2,header=None, names=['Year','Month','Day','Temperature','WindSpeedMPH'])
print(df.head())

# read only 3 rows
df = pd.read_csv('nyc_weather.csv',skiprows=2,header=None, names=['Year','Month','Day','Temperature','WindSpeedMPH'], nrows=3)

# na_values equal to not available, 'n.a'
df = pd.read_csv('nyc_weather.csv',skiprows=2,header=None, names=['Year','Month','Day','Temperature','WindSpeedMPH'], na_values=['n.a', 'not available'])
print(df.head())