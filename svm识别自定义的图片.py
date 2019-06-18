from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import scipy
import cv2
import os

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        return files

def image2Digit(image):
    # 调整为300*300大小
    im_resized = scipy.misc.imresize(image, (300,300))
    # RGB（三维）转为灰度图（一维）
    # im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)

    # 调整为0-16之间（digits训练数据的特征规格）像素值——16/255
    #im_hex = Fraction(16,255) * im_gray
    # 将图片数据反相（digits训练数据的特征规格——黑底白字）
    #im_reverse = 16 - im_hex
    return im_resized.reshape(90000*3)
    # return im_gray.astype(np.int)

def readImage(path):
    namelist=file_name(path)
    imagelist=[]
    for name in namelist:
        imagepath=path+'/'+name
        image = cv2.imread(imagepath)
        imagelist.append(image2Digit(image))
    return imagelist

def run():
    X_train = readImage('container_folder/real_train')
    y_train = [ 1 for i in range(len(X_train))]

    X_test_real_pre = readImage('container_folder/real_test')
    y_test_real_pre = [ -1 for i in range(len(X_test_real_pre))]

    X_test_fake_pre = readImage('container_folder/fake_train')
    y_test_fake_pre = [ 1 for i in range(len(X_test_fake_pre))]

    X_test=X_test_real_pre+X_test_fake_pre
    y_test=y_test_real_pre+y_test_fake_pre



    model = OneClassSVM(gamma='auto')

    model.fit(X_train,y_train)


    ans=y_test-model.predict((X_test))

    i=0
    for a in ans:
        if a==0:
            i=i+1
    print(i/len(ans))

for i in range(3):
    run()
# def run():
#     ocsvm=OneClassSVM()
#     # 读取单张自定义手写数字的图片
#     image = scipy.misc.imread("digit_image/2.png")
#     # 将图片转为digits训练数据的规格——即数据的表征方式要统一
#     im_reverse = image2Digit(image)
#     # 显示图片转换后的像素值
#     print(im_reverse)
#     # 预测
#     result = ocsvm.predict(im_reverse)
#     print(result)