from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from preprocess import preprocess
from transformers import AdamW, get_linear_schedule_with_warmup
from train_test_split import train_val_split
from tokenizer import tokenize


data_path = './data/smileannotationsfinal.csv'

df, label_dict = preprocess(data_path)
df = train_val_split(df)
dataset_train, dataset_val = tokenize(df)


# Reason we use base model is because the architecture is small and hence it's computationally tractable
# BertForSequneceClassification: Bert transformer with a sequnce classification head on the top
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                    num_labels = len(label_dict), # number of output labels
                                    output_attentions = False, # we dont't want model to output attentions
                                    output_hidden_states = False) # # we dont't want model to output final hidden states

batch_size = 32

# Creating training and validation dataloaders
dataloader_train = DataLoader(dataset_train, sampler = RandomSampler(dataset_train), batch_size = batch_size)
dataloader_val = DataLoader(dataset_val, sampler = RandomSampler(dataset_val), batch_size = batch_size)

# Initializing the optimizer
# https://huggingface.co/transformers/main_classes/optimizer_schedules.html
optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

epochs = 10 

# Create a schedule with a learning rate that decreases linearly from the initial lr set 
# in the optimizer to 0, after a warmup period during which it increases linearly from 0 to 
# the initial lr set in the optimizer.
scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps = len(dataloader_train)*epochs, num_warmup_steps = 0)