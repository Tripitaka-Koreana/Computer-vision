cropped_img = img[y1:y2, x1:x2]

mask = np.ones(img.shape[:2], dtype=np.uint8)

# 제외할 영역을 0으로 설정
mask[y1:y2, x1:x2] = 0

# 마스크를 사용하여 특정 영역 제외
excluded_region = cv.bitwise_and(img, img, mask=mask)
