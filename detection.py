
import os
import shutil
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# labelmap= ( 'class1','class2','class3','class4','class5','class6','class7','class8','class9','class10',)
# labelmap = (  # always index 0
#     'bn','he','nm','tb','tk','kn')
labelmap=('e1','e2','e3','e4')
color =((0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),(128,255,0),(255,128,0),(0,128,255),(255,0,128))
# home_path='data/HongKongdatasets/'
home_path='data/TILDAdatasets/'
# home_path='data/HongKong_test_dataAugment/'
# home_path='data/DAGMdatasets_test/'



det_result=home_path+"VOCdevkit/VOC2007/results"
result_floder=home_path+"VOCdevkit/VOC2007/showResult"
path = home_path+'VOCdevkit/VOC2007/showResult'
txtpath=home_path+'VOCdevkit/VOC2007/ImageSets/Main/test.txt'
if not os.path.exists(result_floder):
    os.mkdir(result_floder)
filelist=os.listdir(result_floder)
for f in filelist:
     filepath = os.path.join( result_floder, f )
     if os.path.isfile(filepath):
        os.remove(filepath)


for root, dirs, files in os.walk(home_path+'VOCdevkit/VOC2007/JPEGImages'):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, os.path.join(result_floder,file))
            # print(os.path.join(result_floder,file))
for i in range(len(labelmap)):
   with open(os.path.join(det_result,"det_test_"+str(labelmap[i])+".txt"),'r') as file:
       while 1:
           line = file.readline()
           if not line:
               break
           list=line.split()
           xmin=int(float(list[2]))
           ymin=int(float(list[3]))
           xmax=int(float(list[4]))
           ymax=int(float(list[5]))
           img=cv2.imread(os.path.join(result_floder,list[0]+".jpg"))# may be PNG
           print(os.path.join(result_floder,list[0]+".jpg"),type(img))# may be PNG
           cv2.rectangle(img, ( xmin,ymin), (  xmax,ymax), color[i], 2)
           cv2.rectangle(img, (xmin, ymin-40), (xmin+200, ymin), color[i], thickness=-1)
           img = Image.fromarray(img)
           ft = ImageFont.truetype("arialuni.ttf", 30)
           draw=ImageDraw.Draw(img)
           draw.text((xmin+5,ymin-35),labelmap[i]+":{:.3f}".format(float(list[1])),font=ft,fill="#000000")
           img = np.array(img)
           cv2.imwrite(os.path.join(result_floder,list[0]+".jpg"), img)
           print(os.path.join(result_floder,list[0]+".jpg")+" has detected!")


list=[]
with open(txtpath,'r') as file:
    while(True):
        line=file.readline()
        if not line:
            break
        print(line.split()[0]+".jpg")
        list.append(line.split()[0]+".jpg")
print(len(list))

for files in os.listdir(path):
    if files not in list:
        os.remove(os.path.join(path,files))



