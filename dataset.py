from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["conv", "strat"]
num_classes = len(classes)
#image_size = 64
image_size=600
image_size2=300
num_testdata = 50

X_train = []
X_test  = []
y_train = []
y_test  = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.png")
    
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        #image = image.resize((image_size, image_size2))
        image = image.resize((128, 256))
        data = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        else:

            # angleに代入される値
            # -20
            # -15
            # -10
            #  -5
            # 0
            # 5
            # 10
            # 15
            #for angle in range(0):
                #img_r = image.rotate(angle)
                #data = np.asarray(img_r)
                #X_train.append(data)
                #y_train.append(index)
            img_r = image.rotate(0)
            data = np.asarray(img_r)
            X_train.append(data)
            y_train.append(index)
                # FLIP_LEFT_RIGHT　は 左右反転
            #    img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            #    data = np.asarray(img_trains)
            #    X_train.append(data)
            #    y_train.append(index)

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./conv_strat.npy", xy)
