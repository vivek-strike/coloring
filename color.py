import numpy as np
import cv2
import time
from os.path import splitext, basename, join

class Colorizer:
    def __init__(self, height = 1080, width = 1920):
        (self.height, self.width) = height, width

        self.colorModel = cv2.dnn.readNetFromCaffe("models/colorization_deploy_v2.prototxt",
                                                   caffeModel="models/colorization_release_v2_norebal.caffemodel")
        
        culsterCenters = np.load("models/pts_in_hull.npy") #changed from clusterCenters 
        culsterCenters = culsterCenters.transpose().reshape(2, 313, 1, 1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [culsterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1,313], 2.606, np.float32)]

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width,self.height))

        self.processFrame()
        cv2.imwrite(join("output", basename(imgName)), self.imgFinal)

        cv2.imshow("output", self.imgFinal)


    def processFrame(self):
        imgNormalized = (self.img[:,:,[2,2,2]] * 1.0/255).astype(np.float32)

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:,:,0]

        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224,224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -= 55

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))

        resultResized = cv2.resize(result, (self.width, self.height))

        self.imgOut = np.concatenate((channelL[:,:,np.newaxis], resultResized), axis = 2)
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0, 2)
        self.imgOut = np.array((self.imgOut)*255, dtype = np.uint8)

        self.imgFinal = np.hstack((self.img, self.imgOut))
