'''
有同事问我，对于明显是随机生成的域名字符串，有没有办法识别出来
我造了一些数据，尝试用一个简单的神经网络进行分类，效果不错。准确率96%以上
'''
import torch
import numpy as np
import random
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

batchsz = 100
lr = 0.001
inputsz = 40
epochnm = 20
minbatch = 0
compute = True


# clean the text, only alpha and space is left
def clean(text):
    result=bytes()
    for i in range(len(text)):
        c = chr(text[i])
        if c.isalpha()  or c.isnumeric():
            result = result + text[i:i+1].lower()
        if c == " ":
            result = result+bytes(".", encoding="utf8")
    return result



# 从一本书里生成随机的 域名的正例。这样的域名通常含有英文单词
def get_positive():
    with open("./data/nietzsche.txt", "rb") as f:
        char_artical = f.read()
    char_artical = clean(char_artical)
    total_len = len(char_artical)
    offset = 0
    ret = list()
    while offset < total_len:
        example_len = random.randint(20, inputsz)
        endpos = offset+example_len
        if endpos > total_len:
            endpos = total_len
        substr = char_artical[offset:endpos]
        offset = endpos

        if substr[0:1] == bytes(".", encoding="utf8"):
            substr = substr[1:]
        if substr[-1:] == bytes(".", encoding="utf8"):
            substr = substr[:-1]

        while len(substr) < inputsz:
	    substr = substr + b'\0'

        if offset < 400:
            print(str(substr))
        t = torch.tensor([ int(substr[i]) for i in range(inputsz)])
        ret.append( (t, 1) )
    return ret

# 随机的字符组成的域名，是反例
def get_negtive():
    ret = list()
    chars = "abcedfghijklmnopqrstuvwxyz0123456789.."
    for i in range(25000):
        example_len = random.randint(20, inputsz)
        example = bytes()
        for j in range(example_len):
            pos = random.randint(0, len(chars))
            example = example + bytes(chars[pos:pos+1], encoding="utf8")
        while len(example) < inputsz:
            example = example+b'\0'

        if i < 10:
            print(example)
        t = torch.tensor([int(example[i]) for i in range(inputsz)])

        ret.append( (t, 0) )
    return ret

class myDataset(dataset.Dataset):
    def __init__(self, isTrain=True):
        super(myDataset, self).__init__()
        self.isTrain = isTrain
        poslist = get_positive()
        neglist = get_negtive()
        self.train = list()
        self.test = list()

        for t in poslist:
            r = random.randint(0, 10)
            if r > 8:
                self.test.append(t)
            else:
                self.train.append(t)
        for t in neglist:
            r = random.randint(0, 10)
            if r > 8:
                self.test.append(t)
            else:
                self.train.append(t)
        random.shuffle(self.test)
        random.shuffle(self.train)
        print("test size:", len(self.test))
        print("train size:", len(self.train))




    def __getitem__(self, index):
        if self.isTrain:
            return self.train[index][0].to(dtype=torch.float32)/255, torch.tensor(self.train[index][1], dtype=torch.long)
        else:
            return self.test[index][0].to(dtype=torch.float32) / 255, torch.tensor(self.test[index][1],
                                                                                    dtype=torch.long)



    def __len__(self):
        if self.isTrain:
            return len(self.train)
        else:
            return len(self.test)

set1 = myDataset()
train_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader

set1 = myDataset(False)
test_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader

#简单的卷积网络。没有卷积层的话，准确率可以到90%，加上卷积层可以到96%
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32*inputsz, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = nn.functional.relu(self.conv1(x))
        x = x.reshape(x.shape[0], -1)
        x = F.dropout(x, 0.1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 用测试集对模型进行校验
def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    total = 0
    same = 0
    for inputs, labels in test_data:
        inputs = inputs.to(device="cuda:0")
        labels = labels.to(device="cuda:0")
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1) # type:torch.return_types.max
        same += (y[1] == labels).sum().to(device='cpu').numpy()
        total += batchsz
        if total > 1000:
            break
    return same / total


# 训练
if compute:
    model = MyModel().to("cuda:0")
    trainer = torch.optim.Adam(model.parameters(), lr)
    lossfun = nn.CrossEntropyLoss()
    lossSum = 0

    for e in range(epochnm):
        model.train()

        for inputs, labels in train_data:
            minbatch += 1
            inputs = inputs.to(device="cuda:0")  # type:torch.tensor()
            labels = labels.to(device="cuda:0")
            y = model(inputs)

            L = lossfun(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            lossSum = lossSum + L.to("cpu").data.numpy()
            if minbatch % 200 == 0:
                print(e, " ", minbatch, " ", lossSum / 200)
                lossSum = 0
        print(e, " acc:", test(model, test_data))
        model.train()
    torch.jit.script(model).save("./findRandStr.pt")

// c++ call the trained model:
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main() {

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load("E:\\DeepLearning\\mxnet\\pyproject\\test1\\findRandStr.pt");

  std::cout << "ok\n";
 
  std::vector<torch::jit::IValue> inputs;
  c10::Device dev = c10::Device("cuda:0");
  //inputs.push_back(torch::ones({ 1,40 }).to(dev));

  char domain[] = "prefacesupposing.that.truth             ";
  torch::Tensor t = torch::zeros({ 2, 40 });
  for (int i = 0; i < 40; ++i)
  {
	  t[0][i] = float(domain[i])/255.0;
	  t[1][i] = 1.0;

  }
  std::cout << "t is ready\n";
  inputs.push_back(t.to(dev));
  std::cout << " input is ready\n";
  module.eval();

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "forward() done\n";
  std::cout << output<<"\n";

 }





