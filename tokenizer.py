from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from preprocess import preprocess
from train_test_split import train_val_split

def tokenize(df):
  '''
  Takes in df, encodes tweets and returns dataset objects for both train and validation set

  Parameters:
  df: data frame returned by train_val_split() in train_test_split.py

  Returns:
  dataset_train: dataset object for training set
  dataset_val: dataset_object for validation set
  '''
    
  '''
  ##########################################################################################
  bert-base-uncased
  ------------------

  Pretrained model trained on lower-cased English text using a masked language modeling (MLM) objective.
  This model is uncased: it does not make a difference between english and English.

  Pretrained Bert Model - 12-layer, 768-hidden, 12-heads, 110M parameters. 
  ##########################################################################################
  '''

  '''
  ##########################################################################################
  Tokenizer:
  ----------

  https://huggingface.co/transformers/model_doc/bert.html

  A tokenizer is in charge of preparing the inputs for a model.
  The library comprise tokenizers for all the models (i.e tokenizing based on the vocabulary for the model).

  The base classes PreTrainedTokenizer and PreTrainedTokenizerFast implements the common methods
  for encoding string inputs in model inputs.

  PreTrainedTokenizer implements the main methods for using all the tokenizers:

      - tokenizing (spliting strings in sub-word token strings), converting tokens strings to ids 
        and back, and encoding/decoding (i.e. tokenizing + convert to integers)

      - adding new tokens to the vocabulary in a way that is independant of the underlying structure 
        (BPE, SentencePiece…),

      - managing special tokens like mask, beginning-of-sentence, etc tokens (adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization)
  #########################################################################################
  '''
  # Instantiate a pretrained pytorch model from a pre-trained model configuration.
  # Using pre-trained 'bert-base-uncased' model to encode our dataset. 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)


  '''
  https://huggingface.co/transformers/glossary.html#attention-mask

  BatchEncoding holds the output of the tokenizer’s encoding methods 
  (__call__, encode_plus and batch_encode_plus) and is derived from a Python dictionary. 
  When the tokenizer is a pure python tokenizer, this class behave just like a standard 
  python dictionary and hold the various model inputs computed by these methodes 
  (input_ids, attention_mask…). 

  # batch_encode_plus can encode multiple string parallely

  documentation for __call__() --> https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer

  All the methods return a dictionary containing the encoded sequence or sequence pair and 
  additional information: the mask for sequence classification and the overflowing elements if a 
  max_length is specified.



  With the fields:

  input_ids: list of token ids to be fed to a model

  token_type_ids: list of token type ids to be fed to a model

  attention_mask: list of indices specifying which tokens should be attended to by the model

  overflowing_tokens: list of overflowing tokens sequences if a max length is specified and
                      return_overflowing_tokens=True.

  special_tokens_mask: if adding special tokens, this is a list of [0, 1], with 0 specifying 
                      special added tokens and 1 specifying sequence tokens.


  '''

  # Encoding training and validation dataset
  encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type == 'train'].text.values,
                                                  add_special_tokens = True,
                                                  return_attention_mask = True,
                                                  pad_to_max_length = True,
                                                  max_length = 256,
                                                  return_tensors = 'pt')

  encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type == 'val'].text.values,
                                                  add_special_tokens = True,
                                                  return_attention_mask = True,
                                                  pad_to_max_length = True,
                                                  max_length = 256,
                                                  return_tensors = 'pt')

  # Getting relevant values from the dictionary returned by tokenizer for train dataset
  input_ids_train = encoded_data_train['input_ids'] #(1258,256)
  attention_masks_train = encoded_data_train['attention_mask'] #(1258,256)
  labels_train = torch.tensor(df[df.data_type == 'train'].label.values) #(1258,)                                                

  # Getting relevant values from the dictionary returned by tokenizer for validation dataset
  input_ids_val = encoded_data_val['input_ids'] # (223,256)
  attention_masks_val = encoded_data_val['attention_mask'] # (223, 256)
  labels_val = torch.tensor(df[df.data_type == 'val'].label.values) # (223,)                                               

  # Dataset wrapping tensors.
  # Each sample will be retrieved by indexing tensors along the first dimension.

  # Creating Dataset object for train and validation data
  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

  return dataset_train, dataset_val

if(__name__ == '__main__'):
  data_path = './data/smileannotationsfinal.csv'
  df = train_val_split(preprocess(data_path)[0])
  print(tokenize(df))
  