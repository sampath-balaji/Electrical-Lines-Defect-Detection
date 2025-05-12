import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
# switch Matplotlib to non-interactive (Agg) backend before pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import random
from PIL import ImageFilter
import torchvision.transforms.functional as F

# ----------- CONFIG --------------------
data_dir       = '../final_dataset_split'
batch_size     = 16
num_epochs     = 40
learning_rate  = 1e-4
use_mixed_precision = True
save_dir       = 'saved_models'
viz_dir        = 'visualizations'            # ← new
best_model_path = os.path.join(save_dir, 'best_dino_model.pt')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)          # ← new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------ TRANSFORMS -------------------
def letterbox_blur_jitter(img, size=448, blur_radius_range=(10, 20)):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.LANCZOS)

    background = F.resize(img, (size, size), interpolation=F.InterpolationMode.LANCZOS)
    r = random.uniform(*blur_radius_range)
    background = background.filter(ImageFilter.GaussianBlur(radius=r))

    x1 = (size - new_w) // 2
    y1 = (size - new_h) // 2
    background.paste(resized, (x1, y1))
    return background

data_transforms = {}
for split in ['train','val','test']:
    tlist = [
        transforms.Lambda(lambda img: letterbox_blur_jitter(img, 448, (10,20)))
    ]
    if split == 'train':
        tlist.append(transforms.ColorJitter(0.2,0.2,0.2))
    tlist += [
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ]
    data_transforms[split] = transforms.Compose(tlist)


# ----------- MODEL SETUP --------------------
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val', 'test']
}
class_names   = image_datasets['train'].classes
num_classes   = len(class_names)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
in_features = model.norm.normalized_shape[0]

# freeze everything...
for param in model.parameters():
    param.requires_grad = False

# ...then unfreeze last 4 blocks...
for blk in model.blocks[-4:]:
    for p in blk.parameters():
        p.requires_grad = True

# ...and replace head in one go (with dropout)
model.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
for p in model.head.parameters():
    p.requires_grad = True

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

model = model.to(device)


# Criterion, Optimizer, Scheduler
criterion    = nn.CrossEntropyLoss()
optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler    = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=5, factor=0.5)
scaler       = torch.amp.GradScaler(enabled=use_mixed_precision)


# Create dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
    for x in ['train','val','test']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val','test']}


# ----------- TRAINING FUNCTION --------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', '-'*10)

        for phase in ['train', 'val']:
            model.train() if phase=='train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = outputs.argmax(dim=1)

                    if phase=='train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase=='val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)

        print()

    elapsed = time.time() - since
    print(f'Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load(best_model_path))
    return model


# ----------- VISUALIZATION FUNCTION -----------
def visualize_dataset(model, phase='val', num_images=6, epoch=None):
    """
    Runs inference on `phase` set, plots up to num_images, saves figure to disk.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15,10))

    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            for j in range(inputs.size(0)):
                images_shown += 1
                ax = plt.subplot(2, 3, images_shown)
                ax.axis('off')
                color = 'green' if preds[j]==labels[j] else 'red'
                ax.set_title(f"P:{class_names[preds[j]]}\nT:{class_names[labels[j]]}", color=color)

                img = inputs.cpu().data[j].numpy().transpose(1,2,0)
                img = np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0,1)
                ax.imshow(img)

                if images_shown == num_images:
                    break
            if images_shown == num_images:
                break

    fname = f"{phase}_viz"
    if epoch is not None:
        fname += f"_ep{epoch}"
    fname += ".png"
    save_path = os.path.join(viz_dir, fname)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


# ----------- RUN TRAINING & VISUALIZE ----------------
model = train_model(model, criterion, optimizer_ft, scheduler, num_epochs=num_epochs)
visualize_dataset(model, phase='val', num_images=6, epoch=num_epochs)
visualize_dataset(model, phase='test', num_images=6, epoch=num_epochs)
