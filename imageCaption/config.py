import torch

class Config:
    # Dataset paths
    IMAGES_DIR = 'flickr8k/Images'
    CAPTIONS_FILE = 'flickr8k/captions.txt'

    # Image transforms
    IMAGE_SIZE = 224
    RESIZE_SIZE = 226
    CROP_SIZE = 224

    # Model architecture
    EMBED_SIZE = 400
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT_PROB = 0.3

    # Training settings
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 2
    BATCH_SIZE = 256
    STEP_SIZE = 4
    GAMMA = 0.1
    NUM_WORKERS = 4

    # Frequency threshold for vocabulary
    FREQ_THRESHOLD = 5

    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard settings
    TENSORBOARD_LOG_DIR = "tensorboard_logs/"

    # Model saving
    SAVE_MODEL_PATH = "saved_models/image_captioning_model.pth"
    MODEL_SAVE_INTERVAL = 1