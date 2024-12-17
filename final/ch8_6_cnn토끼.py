import numpy as np
import cv2 as cv
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

#%% ResNet50 Model Load

#백본 모델로 ResNet50 사용
model=ResNet50(weights="imagenet")

img=cv.imread("rabbit.jpg")
x=np.reshape(cv.resize(img, (224,224)),(1, 224,224,3))
x=preprocess_input(x)

preds=model.predict(x) #preds.shape : 1x1000 -> ImageNet이 1000부류를 가지기 때문에 1000개 부류에 대한 확률을 출력
#preds에서 확률이 가장 높은 5개 추출하여 부류 이름과 함께 저장
top5=decode_predictions(preds, top=5)[0] #return list
print("Prediction", top5)

for i in range(5):
    cv.putText(img, top5[i][1]+':'+str(top5[i][2]),(10, 20+i*20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    
cv.imshow("Result", img)

cv.waitKey()
cv.destroyAllWindows()
         

