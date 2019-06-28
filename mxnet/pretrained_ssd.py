from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

########### load pretrained model ##################
#net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
net = model_zoo.ssd_512_resnet50_v1_voc(pretrained=True)

########## get image and preprocess it #############
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
print(im_fname)

# short: Resize image short side to this short and keep aspect ratio.
# details are in https://gluon-cv.mxnet.io/api/data.transforms.html#gluoncv.data.transforms.presets.ssd.load_test
# two  lines below are both ok.
#x, img = data.transforms.presets.ssd.load_test(im_fname,short=512)
x, img = data.transforms.presets.ssd.load_test("e:/DeepLearning/data/test/5.jpg",short=512)
print('Shape of pre-processed image:', x.shape)

######### call the net ############################
class_IDs, scores, bounding_boxes = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()