import pandas as pd

df = pd.read_csv('nyc_weather.csv')
print(df.head())
print(f'Max Temparature { df['Temperature'].max() }')
print(f'Wind Speed {df['WindSpeedMPH'].mean()}')
df.fillna(0, inplace=True)
print(f'Wind Speed {df['WindSpeedMPH'].mean()}')