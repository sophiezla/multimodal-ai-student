import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchinfo import summary

import nltk # Natural Language Toolkit
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence # From the RNN Library
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class Vocabulary:
    def __init__(self, freq_threshold):
        # We need a way to convert from word to index and vice versa 
        self.index_to_word = {0: "<PAD>", 1: "<UNK>", 2: "< SOS >", 3: "<EOS>"}
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1, "< SOS >": 2, "<EOS>": 3}
        self.freq_threshold = freq_threshold 

    def __len__(self):
        return len(self.index_to_word)

    @staticmethod
    def tokenizer(text):
        return [word for word in word_tokenize(text.lower())]

    def build_vocab(self, sentence_list):

        frequencies = {}
        start_idx = 4 # Start index of the vocabulary

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if token not in frequencies:
                    frequencies[token] = 1
                else:
                    frequencies[token] += 1
                if frequencies[token] == self.freq_threshold:
                    self.word_to_index[token] = start_idx
                    self.index_to_word[start_idx] = token
                    start_idx += 1

    def get_indices(self, sentence):
        '''
        Turn a sentence into a list of indices.

        Each word corresponds to an indices in the word_to_index dictionary we have built above 
        If a word does not exist then we return it as an Unkown ("UNK") token
        '''
        tokenized_text = self.tokenizer(sentence)
        return [
            self.word_to_index[word] if word in self.word_to_index else self.word_to_index["<UNK>"]
            for word in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transforms=None, freq_threshold=2):
        self.root_dir = root_dir 
        self.df = pd.read_csv(caption_file)
        self.transforms = transforms # Image transformations

        self.img_path = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.img_path[idx]
        caption = self.captions[idx]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transforms:
            img = self.transforms(img)            

        numericalized_caption = [self.vocab.word_to_index["< SOS >"]]
        numericalized_caption += self.vocab.get_indices(caption)
        numericalized_caption.append(self.vocab.word_to_index["<EOS>"])

        # Image will be a torch tensor because it will be included in our transforms, we need to convert caption
        return img, torch.tensor(numericalized_caption)

def get_loader(root_dir, captions_file, transform, batch_size, shuffle=False, num_workers=0):
    
    dataset = FlickrDataset(root_dir, captions_file, transform)

    pad_idx = dataset.vocab.word_to_index["<PAD>"]

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        pin_memory=True, # Advanced Concept: Data placed in "VIP Section" of computer gets fast-track lane to the GPU when training
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return data_loader, dataset
# This object is to pad captions to ensure they are all the same length
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        captions = [item[1] for item in batch]
        padded_captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return imgs, padded_captions

def caption_image(model, image, vocabulary, max_length=30):
    """
    Generate a caption for a single image using a trained model.
    Args:
        model: The CNN_to_LSTM model
        image: A single image tensor (1, 3, 224, 224)
        vocabulary: The Vocabulary object
        max_length: Max caption length

    Returns:
        List of predicted words
    """
    model.eval()
    result = []

    with torch.no_grad():
        encoder_features = model.encoder(image)

        word_idx = vocabulary.word_to_index["< SOS >"]
        result.append(word_idx)

        inputs = model.decoder.embed(torch.tensor([word_idx], device=image.device)).unsqueeze(0)
        x = torch.cat((encoder_features.unsqueeze(0), inputs), dim=0)

        states = None

        for i in range(max_length - 1):
            if i == 0:
                lstm_out, states = model.decoder.lstm(x)
            else:
                lstm_out, states = model.decoder.lstm(inputs, states)

            output = model.decoder.fc(lstm_out[-1])
            predicted_idx = output.argmax(1).item()
            result.append(predicted_idx)

            if predicted_idx == vocabulary.word_to_index["<EOS>"]:
                break

            inputs = model.decoder.embed(torch.tensor([predicted_idx], device=image.device)).unsqueeze(0)

    # Clean up output
    special_tokens = {"<PAD>", "<UNK>", "< SOS >", "<EOS>"}
    return [vocabulary.index_to_word[idx] for idx in result 
            if vocabulary.index_to_word[idx] not in special_tokens]



'''
Below is backup code for students to use when inference on the hugging face model. Since you can not import 
functions between ipynb files, when testing we will just import this model (this is the same model students will write)'''

class CNN_Encoder(nn.Module): 
    def __init__(self, embed_size):
        super().__init__()

        # Use ResNet18 which is simpler and well-established
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        
        # Add a new FC layer to get the embedding size we want
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

    def forward(self, images):
        features = self.resnet(images)
        # Flatten the feature map/image
        features = self.flatten(features)
        # Pass through our new FC layer
        features = self.fc(features)
        # Apply dropout
        features = self.dropout(features)
        return features

class LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_features, captions, states=None):
        """
        Forward pass that can handle both training (with teacher forcing) and inference
        
        Args:
            encoder_features: Features from the CNN encoder
            captions: Caption tokens for teacher forcing
            states: (h, c) LSTM states (optional, for sequential inference)
            
        Returns:
            outputs: Word predictions at each time step
            states: LSTM hidden states (for inference continuation)
        """
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))
        
        # Append the encoder features as the first "word" in the sequence
        embeddings = torch.cat((encoder_features.unsqueeze(0), embeddings), dim=0)
        
        # Pass through LSTM (returning states for sequential processing)
        lstm_out, states = self.lstm(embeddings, states)
        
        # Get predictions for each word
        outputs = self.fc(lstm_out)
        
        return outputs, states

class CNN_to_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super().__init__()

        self.encoder = CNN_Encoder(embed_size)
        self.decoder = LSTM_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """
        Forward pass for training with teacher forcing
        """
        encoder_features = self.encoder(images)
        
        outputs, _ = self.decoder(encoder_features, captions)
        
        return outputs