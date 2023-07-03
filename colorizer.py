from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

class Colorizer:
    # Colorizer class implementation goes here
    def __init__(self, height=1080, width=1920):
        (self.height, self.width) = height, width

        self.colorModel = cv2.dnn.readNetFromCaffe("coloring/models/colorization_deploy_v2.prototxt",
                                                   caffeModel="coloring/models/colorization_release_v2_norebal.caffemodel")

        culsterCenters = np.load("coloring/models/pts_in_hull.npy")  # changed from clusterCenters
        culsterCenters = culsterCenters.transpose().reshape(2, 313, 1, 1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [culsterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    def processImage(self, img):
        img = cv2.resize(img, (self.width, self.height))

        imgNormalized = (img[:, :, [2, 2, 2]] * 1.0 / 255).astype(np.float32)

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:, :, 0]

        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224, 224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:, :, 0]
        channelLResized -= 55

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))

        resultResized = cv2.resize(result, (self.width, self.height))

        imgOut = np.concatenate((channelL[:, :, np.newaxis], resultResized), axis=2)
        imgOut = np.clip(cv2.cvtColor(imgOut, cv2.COLOR_LAB2BGR), 0, 1)
        imgOut = np.array((imgOut) * 255, dtype=np.uint8)

        imgFinal = np.hstack((img, imgOut))

        return imgFinal

@app.route('/colorize', methods=['POST'])
def colorize_image():
    # Get the image file from the request
    image_file = request.files['image']
 
    # Read the image using OpenCV
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
 
    # Initialize the colorizer
    colorizer = Colorizer(width=640, height=480)

    # Process the image
    colorized_image = colorizer.processImage(img)

    # Convert the colorized image to bytes
    _, colorized_bytes = cv2.imencode('.jpg', colorized_image)

    # Convert the bytes to base64
    colorized_base64 = base64.b64encode(colorized_bytes).decode('utf-8')

    # Prepare the response JSON
    response = {'image': colorized_base64}

    # Return the response JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500)