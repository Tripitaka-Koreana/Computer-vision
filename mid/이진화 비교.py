import cv2 as cv
#%%
gray=cv.imread("book.bmp",0); print(gray.shape)
cv.imshow("Gray", gray); cv.waitKey()

#임계값 70
t, binaryFixed=cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
cv.imshow("BinaryFixed", binaryFixed), cv.waitKey()

#오츄 이진화
t, binaryOtsu=cv.threshold(gray, 70, 255, cv.THRESH_BINARY + cv.THRESH_OTSU); print(t)
cv.imshow("binaryOtsu", binaryOtsu), cv.waitKey()

#적응적 이진화
#지역 블록, 지역 임계값 설정
blockSize=21; C=10
binaryAdaptive=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, C)
cv.imshow("binaryAdaptive", binaryAdaptive), cv.waitKey()
cv.destroyAllWindows()
