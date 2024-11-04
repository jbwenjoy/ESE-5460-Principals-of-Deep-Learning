#!/usr/bin/env python
# coding: utf-8

# In[1]:


dir_root = '.'
# If using google colab
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive/')
    dir_root = '/content/drive/MyDrive/Colab Notebooks/ESE546/hw3'

print(dir_root)


# In[2]:


import requests

# Load the text from a local file
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Load the text from a URL
def load_text_from_url(url):
    response = requests.get(url)
    text = response.text.replace('\r\n', '\n')  # Normalize line endings
    return text

# Count unique characters in the text
def count_unique_chars(text):
    unique_chars = set(text)
    # Num of unique characters
    vocab_size = len(unique_chars)
    return vocab_size, unique_chars

# List of file paths or URLs
local_file_1 = 'pg100.txt'
local_file_2 = 'pg2600.txt'
local_file_3 = 'pg766.txt'
url_file_1 = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'
url_file_2 = 'https://www.gutenberg.org/cache/epub/2600/pg2600.txt'
url_file_3 = 'https://www.gutenberg.org/cache/epub/766/pg766.txt'

if 'google.colab' in str(get_ipython()):
    file_path_list = [url_file_1, url_file_2, url_file_3]
else:
    file_path_list = [local_file_1, local_file_2, local_file_3]
text_list = []
vocab_size_list = []
unique_chars_list = []

for file_path in file_path_list:
    if file_path.startswith('http'):
        print(f'Loading text from URL: {file_path}')
        text = load_text_from_url(file_path)
    else:
        print(f'Loading text from file: {file_path}')
        text = load_text_from_file(file_path)
    vocab_size, unique_chars = count_unique_chars(text)
    text_list.append(text)
    vocab_size_list.append(vocab_size)
    unique_chars_list.append(unique_chars)

print(f'Vocabulary size for each text: {vocab_size_list}')
print(f'Unique characters for each text: {unique_chars_list}')


# In[5]:


# Create a dictionary to map characters to indices and vice-versa
def create_char_mappings(unique_chars):
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return char_to_index, index_to_char

vocab_size, unique_chars = vocab_size_list[0], unique_chars_list[0]
char_to_index, index_to_char = create_char_mappings(unique_chars)
print(f"Character to index mapping for first text: {char_to_index}")
print(f"Index to character mapping for first text: {index_to_char}")


# In[6]:


from tqdm import tqdm
import numpy as np

def generate_sequences_for_transformer(text, char_to_index, seq_length=128, stride=16):
    """
    Generate sequences of indices for transformer training.
    Returns input and target sequences as integer indices instead of one-hot vectors.
    """
    input_sequences = []
    target_sequences = []
    
    for i in tqdm(range(0, len(text) - seq_length - stride, stride), 
                 desc="Generating sequences"):
        # Input sequence
        input_seq = text[i:i + seq_length]
        # Target sequence (shifted by 1 position)
        target_seq = text[i + 1:i + seq_length + 1]
        
        # Convert characters to indices
        input_indices = [char_to_index[c] for c in input_seq]
        target_indices = [char_to_index[c] for c in target_seq]
        
        input_sequences.append(input_indices)
        target_sequences.append(target_indices)
    
    return np.array(input_sequences), np.array(target_sequences)


# In[10]:


import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


# In[12]:


# Data preparation
USE_ALL_BOOKS = False
if 'google.colab' in str(get_ipython()):
    USE_ALL_BOOKS = True

sequence_length = 128
stride = 16

if USE_ALL_BOOKS:
    # Load text and prepare data
    all_texts = ''.join(text_list)
    all_texts = all_texts[:len(all_texts)//10]
    all_texts += all_texts[len(all_texts)//2:len(all_texts)//2 + len(all_texts)//10]
    all_texts += all_texts[len(all_texts)//2 + len(all_texts)//10:]
    vocab_size, unique_chars = count_unique_chars(all_texts)
    char_to_index, index_to_char = create_char_mappings(unique_chars)
    print(f"Vocabulary size: {vocab_size}")
    input_seqs, target_seqs = generate_sequences_for_transformer(
        all_texts, char_to_index, seq_length=sequence_length, stride=stride
    )
else:
    text = text_list[0]
    text = text[:len(text)//10]  # Using first 1/100 of the text
    vocab_size, unique_chars = count_unique_chars(text)
    char_to_index, index_to_char = create_char_mappings(unique_chars)
    print(f"Vocabulary size: {vocab_size}")
    input_seqs, target_seqs = generate_sequences_for_transformer(
        text, char_to_index, seq_length=sequence_length, stride=stride
    )

# Convert to PyTorch tensors
input_seqs = torch.tensor(input_seqs, dtype=torch.long).to(device)
target_seqs = torch.tensor(target_seqs, dtype=torch.long).to(device)

# Create datasets
dataset = TensorDataset(input_seqs, target_seqs)
train_size = int(0.8 * len(dataset))
train_dataset = TensorDataset(input_seqs[:train_size], target_seqs[:train_size])
val_dataset = TensorDataset(input_seqs[train_size:], target_seqs[train_size:])

print("Input sequences shape:", input_seqs.shape)
print("Target sequences shape:", target_seqs.shape)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# In[ ]:





# In[ ]:





# In[23]:


import matplotlib.pyplot as plt

# Save the accuracies and errors and .npy file
np.save(f'{dir_root}/training_losses.npy', training_losses)
np.save(f'{dir_root}/validation_losses.npy', validation_losses)
np.save(f'{dir_root}/training_accuracies.npy', training_accuracies)
np.save(f'{dir_root}/validation_accuracies.npy', validation_accuracies)
np.save(f'{dir_root}/update_counts.npy', update_counts)

# Calculate errors: error = 1 - accuracy
training_errors = [1 - acc for acc in training_accuracies]
validation_errors = [1 - acc for acc in validation_accuracies]

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(update_counts, training_losses, label='Training Loss')
plt.plot(range(0, len(validation_losses) * 1000, 1000), validation_losses, label='Validation Loss') 
plt.xlabel('Weight Updates')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Weight Updates')
plt.legend()

# Plot training and validation error
plt.figure(figsize=(10, 5))
plt.plot(update_counts, training_errors, label='Training Error')
plt.plot(range(0, len(validation_errors) * 1000, 1000), validation_errors, label='Validation Error')  
plt.xlabel('Weight Updates')
plt.ylabel('Error')
plt.title('Training and Validation Error vs. Weight Updates')
plt.legend()

plt.show()


# In[17]:


# Save the model
model_path = f'{dir_root}/char_rnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# In[18]:


# Load the model
model = CharRNN(vocab_size, vocab_size, hidden_size).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully")


# In[ ]:


# Shut down if it's google colab
# First sleep for a while so that changes to the notebook are saved
import time
time.sleep(10)

if 'google.colab' in str(get_ipython()):
    from google.colab import runtime
    runtime.unassign()

