from sklearn.model_selection import train_test_split
from preprocess import preprocess

data_path = './data/smileannotationsfinal.csv'

# Preproccesed data
df = preprocess(data_path)

class_counts = df.category.value_counts()

# As we see that dataset is imbalanced i.e. number of samples for different class is not balanced,
# therefore we need startified train test split i.e. doing 85/15 % split for every class rather than 
# on the entire dataset. If strartiified split isn't done, there might be a chance that no samples 
# are present for the class having the least number of samples in the training dataset and our model 
# won't get trained for that class.
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size = 0.15, random_state = 17, stratify = df.label.values)

# Appending a column named 'data_type' whose value is either train/val depending on the sample
df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

print(df.groupby(['category', 'label', 'data_type']).count())