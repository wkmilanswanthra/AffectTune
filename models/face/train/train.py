import torch, torchvision as tv, torch.nn as nn, torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader
import onnx, os

EMOTIONS = 7
BATCH=32; EPOCHS=3; LR=1e-3; IMG=224

train_tf = T.Compose([T.Resize((IMG,IMG)), T.RandomHorizontalFlip(), T.ToTensor()])
val_tf   = T.Compose([T.Resize((IMG,IMG)), T.ToTensor()])

train_ds = tv.datasets.ImageFolder("data/processed/face/train", transform=train_tf)
val_ds   = tv.datasets.ImageFolder("data/processed/face/val",   transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,  num_workers=4)

model = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.last_channel, EMOTIONS)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
opt = optim.AdamW(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

for ep in range(EPOCHS):
    model.train()
    for x,y in train_dl:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
    # quick val
    model.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in val_dl:
            x,y=x.to(device),y.to(device)
            pred=model(x).argmax(1); corr+= (pred==y).sum().item(); tot+=y.numel()
    print(f"epoch {ep+1}: val_acc={corr/tot:.3f}")

os.makedirs("models/face", exist_ok=True)
torch.save(model.state_dict(), "models/face/mobilenet_v2.pt")

# ONNX export
dummy = torch.randn(1,3,IMG,IMG, device=device)
torch.onnx.export(model, dummy, "models/face/mobilenet_v2.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
print("Exported ONNX -> models/face/mobilenet_v2.onnx")
