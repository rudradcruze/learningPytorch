import torch
from model import EncoderDecoder
from dataset import FlickrDataset
from config import Config
import torchvision.transforms as T
from dataset import get_data_loader

def test_model():
    # Load the saved model
    transforms = T.Compose([
        T.Resize(226),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(
        root_dir=Config.IMAGES_DIR,
        caption_file=Config.CAPTIONS_FILE,
        transform=transforms
    )
    vocab_size = len(dataset.vocab)
    model = EncoderDecoder(Config.EMBED_SIZE, Config.HIDDEN_SIZE, vocab_size, Config.NUM_LAYERS).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.SAVE_MODEL_PATH))
    model.eval()

    # Display a test example with predicted caption
    data_loader = get_data_loader(dataset, batch_size=1, shuffle=False)
    img, _ = next(iter(data_loader))
    img = img.to(Config.DEVICE)

    features = model.encoder(img)
    generated_caption = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)

    print("Generated Caption: ", " ".join(generated_caption))

if __name__ == '__main__':
    test_model()
