import pandas as pd
# df = pd.read_csv('nyc_weather.csv') # instead of this we use inline data
weather_data = {
  'day': ['1/1/2017', '1/2/2017', '1/3/2017', '1/5/2017', '1/6/2017', '1/7/2017'],
  'temparature': [32, 35, 28, 24, 32, 31],
  'windspeed': [6, 7, 2, 7, 7, 4],
  'event': ['Rain', 'Sunny', 'Snow', 'Snow', 'Rain', 'Sunny']
}
df = pd.DataFrame(weather_data)
print(df)
rows, columns = df.shape
print(f"Rows: {rows}, Columns: {columns}")
print(df.head(2)) # print first 2 rows
print(df.tail(2)) # print last 2 rows
print(df.head())
print(df.tail())

print(df[2:5]) # print rows from 2 to 5

print(df.columns) # print column names

print(df.day)
print(df.event)
print('Type of data frame is: ', type(df))

print(df[['event', 'temparature', 'day']]) # print multiple columns

print(f'Max Temparature is {df['temparature'].max()}') # print max temp
print(f'Min Temparature is {df['temparature'].min()}') # print max temp
print(f'Mean Temparature is {df['temparature'].mean()}') # print max tempprint(f'Max Temparature is {df['temparature'].max()}') # print max temprint(f'Max Temparature is {df['temparature'].median()}') # print max tempp
#instread of above
print(df.describe())

# Conditional statements

# Temp >=32
print(df[df['temparature'] >= 32])
# or

print(df.query('temparature >= 32'))

# or

print(df[df['temparature'].between(20, 30)])

# or

print(df[df.temparature>=32]) # print max temp)

# day when temp is max

print(df[['day', 'temparature']][df.temparature==df['temparature'].max()])