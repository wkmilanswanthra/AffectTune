import torch, torch.nn as nn, torch.optim as optim, torchaudio as ta, os, glob
import librosa, numpy as np

CLASSES = ["happy","sad","neutral","angry","fear","surprise","disgust"]  # map your subset
IDX = {c:i for i,c in enumerate(CLASSES)}
SR=16000; DUR=2.0; N_MELS=64

def load_mel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = y[:int(SR*DUR)]
    M = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    L = librosa.power_to_db(M)
    return L.astype(np.float32)  # (n_mels, t)

def load_split(split):
    X,Y=[],[]
    for c in CLASSES:
        for p in glob.glob(f"data/processed/voice/{split}/{c}/*.wav"):
            X.append(load_mel(p))
            Y.append(IDX[c])
    X = np.stack([np.expand_dims(x,0) for x in X],0) # (B,1,n_mels,t)
    return torch.from_numpy(X), torch.tensor(Y)

class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(32, n_classes)
        )
    def forward(self,x): return self.net(x)

train_x, train_y = load_split("train")
val_x, val_y = load_split("val")

device = "cuda" if torch.cuda.is_available() else "cpu"
model=TinyCNN(len(CLASSES)).to(device)
opt=optim.AdamW(model.parameters(), lr=1e-3)
crit=nn.CrossEntropyLoss()

for ep in range(4):
    model.train()
    idx = torch.randperm(train_x.size(0))
    for i in range(0, len(idx), 32):
        b = idx[i:i+32]
        x,y = train_x[b].to(device), train_y[b].to(device)
        opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(val_x.to(device)).argmax(1).cpu()
        acc = (pred==val_y).float().mean().item()
    print("ep", ep+1, "val_acc", round(acc,3))

os.makedirs("models/voice", exist_ok=True)
torch.save(model.state_dict(), "models/voice/tinycnn.pt")
dummy = torch.randn(1,1,N_MELS, 128).to(device)  # t dimension approx
torch.onnx.export(model, dummy, "models/voice/tinycnn.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
print("Exported ONNX voice -> models/voice/tinycnn.onnx")
