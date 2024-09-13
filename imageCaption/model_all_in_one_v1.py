import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# Set up the figure size for plotting
plt.rcParams['figure.figsize'] = [15, 10]

# Dataset paths
images_dir = 'flickr8k/Images'
captions_file = 'flickr8k/captions.txt'

# Display some example images and captions
def show_image(img, title=None):
    """Display an image."""
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Load and display an image with its caption
def display_sample_image(data_desc):
    plt.figure(figsize=(6,6))
    img_path = os.path.join(images_dir, "667626_18933d713e.jpg")
    img = plt.imread(img_path)
    plt.imshow(img)
    print(data_desc["caption"].iloc[0])

# Define the Vocabulary class
class Vocabulary:
    spacy_eng = spacy.load("en_core_web_sm")

    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in Vocabulary.spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

# Define the FlickrDataset class
class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        if self.transform:
            img = self.transform(img)

        caption_vec = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        return img, torch.tensor(caption_vec)

# Define the CapsCollate class
class CapsCollate:
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets

# Define the function to get the DataLoader
def get_data_loader(dataset, batch_size, shuffle=False, num_workers=1):
    pad_idx = dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

# Define the EncoderCNN class
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# Define the DecoderRNN class
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(x)
        return x

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        batch_size = inputs.size(0)
        captions = []
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        return [vocab.itos[idx] for idx in captions]

# Define the EncoderDecoder class
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

def main():
    # Set hyperparameters and initialize the model
    embed_size = 400
    hidden_size = 512
    vocab_size = None
    num_layers = 2
    learning_rate = 0.0001
    num_epochs = 20
    BATCH_SIZE = 256
    NUM_WORKER = 4

    # Define transformations
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Initialize the dataset and dataloader
    dataset = FlickrDataset(
        root_dir=images_dir,
        caption_file=captions_file,
        transform=transforms
    )

    # Check if dataset is correctly initialized and has vocab
    print("Dataset vocab size:", len(dataset.vocab))

    data_loader = get_data_loader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True
    )

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize model with correct vocab size
    vocab_size = len(dataset.vocab)
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Train the model
    for epoch in range(num_epochs):
        for idx, (image, captions) in enumerate(data_loader):
            image, captions = image.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(image, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            if (idx + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")
                model.eval()
                with torch.no_grad():
                    dataiter = iter(data_loader)
                    img, _ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)
                    caption = ' '.join(caps)
                    show_image(img[0], title=caption)
                model.train()
        scheduler.step()

if __name__ == '__main__':
    main()
