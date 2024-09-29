# Read / Load CSV

pd.read_csv('data.csv')

# Read / Load Excel
pdf.read_excel('data.xlsx', 'Sheet1')

# From dictionary
temp_dictionary = [{  }] # key value pair of objects

df = pd.DataFrame(temp_dictionary)

df.to_csv('data.csv', index=False)

# List with Tuple

weather_data = [('New York', 20, 15), ('Los Angeles', 25, 18), ('Chicago', 22, 19)]

df = pd.DataFrame(weather_data, columns=['City', 'Temperature', 'Humidity'])
