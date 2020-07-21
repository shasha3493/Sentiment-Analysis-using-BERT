import pandas as pd

# Reading the data to a dataframe with column names as id, text and category
df = pd.read_csv('./data/smileannotationsfinal.csv', names = ['id', 'text', 'category'])

# Setting 'id' as index of the data frame
df.set_index('id', inplace = True)

print('Printing first 5 entries of the dataset...')
print(df.head())
print()

# Different emotions and their count in the dataset to ensure class balance
print('Different categories and their counts...')
print(df.category.value_counts())
print()

# Removing ths samples that have multiple emotions and have category as 'nocode'
df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']

print("Different categories and their counts after removing smaples with multiple emotions and emotion as 'nocode'...")
print(df.category.value_counts())
print()

# Creating a category to id dictionary
possible_labels = df.category.unique()
label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}

# Adding a new column 'label' to the dataframe which contains he id for the category
df['label'] = df.category.replace(label_dict)
print('Printing 5 entries of the processed dataset...')
print(df.head())
print()

