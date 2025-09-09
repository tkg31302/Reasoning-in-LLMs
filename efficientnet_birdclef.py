import os, random, glob, torch, torchaudio, torchvision, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, Compose
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

AUDIO_DIR = "/kaggle/input/birdclef-2025/train_audio"
SAMPLE_RATE = 32000
CLIP_SECONDS = 5
NUM_MELS = 128
BATCH_SIZE = 16
N_TRAIN = 512

class BirdDataset(Dataset):
    def __init__(self, files, label_map):
        self.files = files
        self.label_map = label_map
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=NUM_MELS,
            f_min=20,
            f_max=16000
        )
        self.resize = Resize((224, 224))
        self.norm = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0, keepdim=True)
        target_len = SAMPLE_RATE * CLIP_SECONDS
        if wav.shape[1] < target_len:
            pad_len = target_len - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        else:
            wav = wav[:, :target_len]
        mel = self.melspec(wav)
        mel = torch.log1p(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        img = torch.repeat_interleave(mel, repeats=3, dim=0)
        img = self.resize(img)
        img = self.norm(img)
        species = os.path.basename(os.path.dirname(path))
        label = self.label_map[species]
        return img, label

def get_files_and_labels(root):
    files = glob.glob(os.path.join(root, "*", "*.ogg"))
    species = sorted({os.path.basename(os.path.dirname(f)) for f in files})
    label_map = {sp: i for i, sp in enumerate(species)}
    random.shuffle(files)
    return files, label_map

files, label_map = get_files_and_labels(AUDIO_DIR)
num_species = len(label_map)
train_files = files[:N_TRAIN]
val_files = files[N_TRAIN:int(1.2 * N_TRAIN)]

train_ds = BirdDataset(train_files, label_map)
val_ds = BirdDataset(val_files, label_map)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_species)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def run_epoch(dl, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

train_loss, train_acc = run_epoch(train_dl, train=True)
val_loss, val_acc = run_epoch(val_dl, train=False)
print(f"Epoch 1 | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}")
torch.save(model.state_dict(), "efficientnet_birdclef_epoch1.pth")
