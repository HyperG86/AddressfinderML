!pip install transformers
!pip install torch
!pip install pandas
!pip install sklearn
!pip install accelerate
!pip install transformers[torch]
import pandas as pd

# Assuming your file is named 'Train_Dataset.xlsx'
df = pd.read_excel('C:\Users\Gayan.waidyasekera\Documents\Work\Calculations\Test Data\ML\Train_Dataset.xlsx')

df['Biiling Address'] = df['Biiling Address'].fillna('').astype(str)
df['Biiling Address1'] = df['Biiling Address1'].fillna('').astype(str)
df['Biiling Address2'] = df['Biiling Address2'].fillna('').astype(str)
df['Billing Post Code'] = df['Billing Post Code'].fillna('').astype(str)
df['Matched Property Address1'] = df['Matched Property Address1'].fillna('').astype(str)
df['Matched Property Address2'] = df['Matched Property Address2'].fillna('').astype(str)
df['Matched Property Post Code'] = df['Matched Property Post Code'].fillna('').astype(str)

df['text'] = df['Biiling Address'] + ' ' + \
             df['Biiling Address1'] + ' ' + \
             df['Biiling Address2'] + ' ' + \
             df['Billing Post Code'] + ' ' + \
             df['Matched Property Address1'] + ' ' + \
             df['Matched Property Address2'] + ' ' + \
             df['Matched Property Post Code']
df['text'] = df['Biiling Address'] + ' ' + df['Biiling Address1'] + ' ' + df['Biiling Address2'] + ' ' + df['Billing Post Code'].astype(str) + ' ' + df['Matched Property Address1'] + ' ' + df['Matched Property Address2'] + ' ' + df['Matched Property Post Code'].astype(str)
from sklearn.model_selection import train_test_split

train_text, test_text, train_labels, test_labels = train_test_split(df['text'], df['Is_Match'], random_state=42, test_size=0.2)
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_text), truncation=True, padding=True)
test_encodings = tokenizer(list(test_text), truncation=True, padding=True)
import torch

class AddressDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AddressDataset(train_encodings, list(train_labels))
test_dataset = AddressDataset(test_encodings, list(test_labels))
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Create trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

# Train the model
trainer.train()
trainer.evaluate()
