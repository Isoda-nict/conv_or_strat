#作ったパラメータを利用してPPIが層状性か対流性かを判別してそれぞれのディレクトリに振り分ける
import keras
import sys, os, glob
import numpy as np
from keras.models import load_model
from PIL import Image

imsize = (128, 256)
testpic = ".20150621-153013.all_raw.dat0.png"　#PPI画像のパスと名前を指定
keras_param="./cnn.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

if __name__ == "__main__":

    model = load_model(keras_param)
    
    classes = "conv"
    classes = "strat"#どちらかを設定
    
    photos_dir = "./" + classes+"/"
    files = glob.glob(photos_dir + "/*.png")
    c=0
    for i, file in enumerate(files):
        img=load_image(file)
        prd=model.predict(np.array([img]))
        prelabel=np.argmax(prd, axis=1)
        print(prd)
        #対流性降水と層状性降水をそれぞれのディレクトリに振り分ける
        if prelabel == 0:
            print(os.path.basename(file)+"対流性")
            c=c+1
        if prelabel == 1:
            print(os.path.basename(file)+"層状性")
            #c=c+1
    print(c)
    
