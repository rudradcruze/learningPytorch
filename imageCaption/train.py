import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from config import Config
import torchvision.transforms as T

from model import EncoderDecoder
from dataset import FlickrDataset, get_data_loader
from config import *


def train_model():
    writer = SummaryWriter(log_dir=Config.TENSORBOARD_LOG_DIR)

    # Dataset and Dataloader
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(
        root_dir=Config.IMAGES_DIR,
        caption_file=Config.CAPTIONS_FILE,
        transform=transforms
    )

    data_loader = get_data_loader(dataset, Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    vocab_size = len(dataset.vocab)

    # Initialize model, loss, optimizer
    model = EncoderDecoder(Config.EMBED_SIZE, Config.HIDDEN_SIZE, vocab_size, Config.NUM_LAYERS).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    print("Starting Training...")

    for epoch in range(Config.NUM_EPOCHS):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for idx, (images, captions) in enumerate(tqdm(data_loader)):
            images, captions = images.to(Config.DEVICE), captions.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images, captions)

            # Calculate loss
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy for the current batch
            predicted = outputs.argmax(dim=2)
            correct_predictions += (predicted == captions).sum().item()
            total_predictions += captions.ne(dataset.vocab.stoi["<PAD>"]).sum().item()

            if (idx + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Step [{idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        epoch_loss /= len(data_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Log loss and accuracy to TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        # Step the scheduler
        scheduler.step()

        # Save the model
        torch.save(model.state_dict(), Config.SAVE_MODEL_PATH)

    writer.close()


if __name__ == '__main__':
    train_model()
