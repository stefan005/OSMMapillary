import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

data_dir = 'ressources/mapillary_raw/validated'
test_dir = 'ressources/mapillary_raw/test'

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def load_split_train_test(datadir, testdir="", valid_size=.2):

    train_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                                            ])
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                                            ])
    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    if testdir != "":
        test_data = datasets.ImageFolder(testdir,
                                         transform=test_transforms)
        weights = make_weights_for_balanced_classes(
            train_data.imgs, len(train_data.classes))
        weights = torch.DoubleTensor(weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(train_data))
        test_sampler = torch.utils.data.sampler.RandomSampler(test_data)
    else:
        test_data = datasets.ImageFolder(datadir,
                                         transform=test_transforms)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=64)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, test_dir)
print(trainloader.dataset.classes)
print(testloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
model = models.resnet50(pretrained=True)
# print(model)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, len(trainloader.dataset.classes)),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
#torch.load(model, 'test-model.pth')
model.to(device)

epochs = 5
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
starttime = datetime.datetime.now()
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            cmt = torch.zeros(len(trainloader.dataset.classes), len(
                trainloader.dataset.classes), dtype=torch.int64)
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    stacked = torch.stack(
                        (labels.reshape((-1, 1)), top_class), dim=1)

                    for p in stacked:
                        tl, pl = p.tolist()
                        cmt[tl, pl] = cmt[tl, pl] + 1

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print(f"now: {datetime.datetime.now().time()} "
                  f"duration: {datetime.datetime.now()-starttime} "
                  f"\nEpoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}"
                  f"\nConfusion Matrix:\n {cmt}"
                  f"\nclasses: {trainloader.dataset.classes}\n\n")
            running_loss = 0
            model.train()
torch.save(model, 'test-model.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig("train_val_plot")
