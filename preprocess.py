import pandas as pd


def preprocess(data_path = './data/smileannotationsfinal.csv'):

    # Reading the data to a dataframe with column names as id, text and category
    df = pd.read_csv(data_path, names = ['id', 'text', 'category'])

    # Setting 'id' as index of the data frame
    df.set_index('id', inplace = True)

    # Removing ths samples that have multiple emotions and have category as 'nocode'
    df = df[~df.category.str.contains('\|')]
    df = df[df.category != 'nocode']

    # Creating a category to id dictionary
    possible_labels = df.category.unique()
    label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}

    # Adding a new column 'label' to the dataframe which contains the id for the category
    df['label'] = df.category.replace(label_dict)
    return df

if (__name__ == '__main__'):
    print(preprocess())

