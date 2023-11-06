import pandas as pd

# Path of the file to read
iowa_file_path = 'Introduction\data\_train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path) 

# print a summary of the data in Melbourne data
print(home_data.describe())
