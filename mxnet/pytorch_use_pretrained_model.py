'''
use pretrained model to do picture Semantic Segmentation
'''
import torch
import torchvision
import torch.utils
import torch.utils.model_zoo
import torch.hub
import numpy as np
import torch.optim
import matplotlib.pyplot as plt
import cv2

batchsz = 1
cropsz = 200


def test(model):
    model.eval()
    with torch.no_grad(): # 节约内存考虑，关闭梯度
        images = torch.zeros((batchsz, 3, 200, 200)).to("cuda:0")
        img = cv2.imread("e://deeplab1.png") # type:np.ndarray
        img2 = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img2[i,j,0] = img[i,j,2]
                img2[i, j, 2] = img[i, j, 0]
                img2[i, j, 1] = img[i, j, 1]
        img = img2
        img = img / 255
        img = cv2.resize(img, (cropsz, cropsz))
        img = img.reshape(cropsz*cropsz, -1)
        img = img.transpose((1,0))
        img = img.reshape(img.shape[0], cropsz, -1)
        img = torch.tensor(img, dtype=torch.float32)
        f = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images[0] = f(img).to("cuda:0")
        yy = model(images)
        yy = yy['out'].to("cpu")
        # shape: batchsz, 21, 40000
        yy = yy.reshape(yy.shape[0], yy.shape[1], -1)  # type:torch.Tensor
        fig = plt.figure(figsize=(5, 5))  # width,height
        for i in range(batchsz):
            pic = (img*255).to(dtype=torch.uint8)  # type:torch.Tensor
            pic = pic.reshape((3, -1))
            pic = pic.transpose(1, 0)
            pic = pic.reshape((cropsz, cropsz, 3))

            plt.subplot(2, batchsz, i + 1)
            plt.imshow(pic)
            plt.axis('off')

            y = yy[i]  # shape:batchsz, 21, 40000  -> 2, 40000
            y = y.transpose(1, 0)
            y = y.argmax(1)

            #np.set_printoptions(threshold=40000)
            #print(y.reshape((200,200)).numpy())
            print("catagories:", set(y.numpy()))

            y = y * 255

            pic = torch.zeros((3, cropsz * cropsz), dtype=torch.uint8)

            pic[0] = y
            pic[1] = y
            pic[2] = y
            pic = pic.transpose(1, 0)
            pic = pic.reshape((cropsz, cropsz, 3))
            plt.subplot(2, batchsz, i + batchsz + 1)
            plt.imshow(pic)
            plt.axis('off')

            plt.show()
            fig.savefig("./tmp.png")


model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True, force_reload=False).to("cuda:0")
test(model)
