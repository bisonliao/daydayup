import torch
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
import PIL.Image
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch.nn as nn

epoches = 10
batchsz = 10000
lr = 0.5
compute = True
file = './data/pytorch_softmax_reg.pk'


transform = transforms.Compose(
    [transforms.ToTensor()])

set1 = tv.datasets.MNIST("./data", True, download=True, transform=transform)
train_data = dataloader.DataLoader(set1, batchsz, True)# type:dataloader.DataLoader

set1 = tv.datasets.MNIST("./data", False, download=True, transform=transform)
test_data = dataloader.DataLoader(set1, batchsz, True)# type:dataloader.DataLoader

def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    for images, labels in test_data:
        images = images.reshape(batchsz, -1).to(device="cuda:0")
        labels = labels.to(device="cuda:0")
        y = model(images)
        y = s(y)
        y = torch.max(y, 1) # type:torch.return_types.max
        same = (y[1] == labels).sum()
        return same.to(device='cpu').numpy() / batchsz



model = nn.Linear(28*28, 10).to(device="cuda:0")

trainer = torch.optim.SGD(model.parameters(), lr)
lossfun = nn.CrossEntropyLoss()



if compute:
    for e in range(epoches):
        model.train()
        for images, labels in train_data:
            images = images.reshape(batchsz, -1).to(device="cuda:0")
            labels = labels.to(device="cuda:0")
            y = model(images)
            trainer.zero_grad()
            L = lossfun(y, labels)
            L.backward()
            trainer.step()
        print("ep:", e, " loss:", L.to("cpu").data.numpy(), " test acc:", test(model, test_data))
    torch.save(model.state_dict(), file)
else:
    model = nn.Linear(28*28, 10).to(device="cuda:0")
    model.load_state_dict(torch.load(file))
    print(test(model, test_data))




