from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from preprocess import preprocess
from transformers import AdamW, get_linear_schedule_with_warmup
from train_test_split import train_val_split
from tokenizer import tokenize
import numpy as np
from sklearn.metrics import f1_score
import random
import torch
from tqdm import tqdm


data_path = './data/smileannotationsfinal.csv'

# Preprocess the data
df, label_dict = preprocess(data_path)
# Train - Validation split
df = train_val_split(df)
# Creating dataset objects
dataset_train, dataset_val = tokenize(df)


# Reason we use base model is because the architecture is small and hence it's computationally tractable
# BertForSequneceClassification: Bert transformer with a sequnce classification head on the top
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                    num_labels = len(label_dict), # number of output labels. This controls the number of nodes in the output layer
                                    output_attentions = False, # we dont't want model to output attentions
                                    output_hidden_states = False) # # we dont't want model to output final hidden states
                                    # Else the output will be a tuple of dim 3 where 3rd dim will be the final hidden states

batch_size = 2 

# Creating training and validation dataloaders
dataloader_train = DataLoader(dataset_train, sampler = RandomSampler(dataset_train), batch_size = batch_size)
dataloader_val = DataLoader(dataset_val, sampler = RandomSampler(dataset_val), batch_size = batch_size)

# Initializing the optimizer
# https://huggingface.co/transformers/main_classes/optimizer_schedules.html
# lr is small as fine tuning
optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

epochs = 10 

# Create a schedule with a learning rate that decreases linearly from the initial lr set 
# in the optimizer to 0, after a warmup period during which it increases linearly from 0 to 
# the initial lr set in the optimizer.
scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps = len(dataloader_train)*epochs, num_warmup_steps = 0)

# Defining performance metrics
def f1_score_func(preds, labels):
    '''
    Calculates the weighted F1-score

    Parameters:
    preds: np array containing the logit scores (batch_size*len(dataloader), 6)
    labels: np array containing the true labels (batch_size*len(dataloader),) 

    Returns:
    Weighted F1 - score
    '''
    # Pred labels
    preds_flat = np.argmax(preds, axis = 1).flatten()

    # True Labels
    labels_flat = labels.flatten()

    # This parameter is required for multiclass/multilabel targets. 
    # If None, the scores for each class are returned. Otherwise, 
    # this determines the type of averaging performed on the data

    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by 
    # support (the number of true instances for each label). This alters 
    # ‘macro’ to account for label imbalance; it can result in an F-score that 
    # is not between precision and recall.
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels):
    '''
    Prints Accuracy per class

    Parameters:
    preds: np array containing the logit scores (batch_size*len(dataloader), 6)
    labels: np array containing the true labels (batch_size*len(dataloader),) 

    Returns:
    None
    '''

    # id2label dictionary 
    label_dict_inverse = {v:k for k,v in label_dict.items()}
    # Predicted labels
    preds_flat = np.argmax(preds, axis = 1).flatten()
    # True labels
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_pred = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print('Class: {} '.format(label_dict_inverse[label]))
        print('Accuracy: {}/{} = {}'.format(len(y_pred[y_pred == label]), len(y_pred), len(y_pred[y_pred == label])/len(y_pred)))
        print('#####################')


# Setting the seed
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate(dataloader_val):
    '''
    Validation Loop
    '''

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

# Training
best_f1 = 0
for epoch in tqdm(range(1, epochs+1)):

    model.train()
    loss_train_total = 0

    # https://github.com/tqdm/tqdm#parameters
    progress_bar = tqdm(dataloader_train, desc = 'Epoch {}'.format(epoch), leave = False, disable = False)

    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
                  
        # Inputs to the model: https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
        outputs = model(**inputs) # Returns a tuple of loss and the logits        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{}'.format(loss.item()/len(batch))})


    tqdm.write('\nEpoch {}'.format(epoch))

    loss_train_avg = loss_train_total/len(dataloader_train)

    tqdm.write('Training Loss: {}'.format(loss_train_avg))

    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)

    tqdm.write('Validation Loss: {}'.format(val_loss))
    tqdm.write('F1 score(weighted): {}'.format(val_f1))

    if val_f1 > best_f1:
        best_f1 = val_f1
        print('Saving the model...')
        torch.save(model.state_dict(),'./Models/best_model.model')

# Evaluation

# Downloading the pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                    num_labels = len(label_dict), # number of output labels
                                    output_attentions = False, # we dont't want model to output attentions
                                    output_hidden_states = False) # # we dont't want model to output final hidden states
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Loading the weights
model.load_state_dict(torch.load('./Models/best_model.model', map_location = torch.device(device)))

_, predictions, true_vals = evaluate(dataloader_val)

accuracy_per_class(predictions, true_vals)