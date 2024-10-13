#이 코드는 다양한 numpy 배열을 생성하고, 정렬, 최대값 및 최소값 계산 등의 기본적인 배열 연산을 수행합니다.

import numpy as np

# 1D numpy 배열을 생성하고 내용과 형태를 출력
a = np.array([4, 5, 0, 1, 2, 3, 6, 7, 8, 9, 10, 11])
print(a)  # 원래 배열 출력
print(a.shape)  # 배열의 형태(요소 개수) 출력

# 배열을 제자리에서 정렬하고 정렬된 배열 출력
a.sort()
print(a)  # 정렬된 배열 출력

# sort 메서드에 대한 도움말 문서 표시
help(a.sort)

# 실수형 숫자를 가진 또 다른 1D numpy 배열 생성
b = np.array([-4.3, -3.3, 12.9, 8.99])
print(b)  # 원래 배열 출력
b.sort()  # 배열을 제자리에서 정렬
print(b) 

# 문자열을 가진 1D numpy 배열 생성
c = np.array(['one', 'two', 'three', 'four'])
c.sort()  # 문자열 배열을 제자리에서 정렬
print(c)

# 2D numpy 배열 생성
d = np.array([[2, 3, 4], [6, 5, 1]])
d.sort()  # 2D 배열의 각 행을 제자리에서 정렬
print(d)

# 2D 배열을 열 방향으로 정렬 (axis=0)
d.sort(axis=0)
print(d)  

# 2D 배열을 행 방향으로 정렬 (axis=1)
d.sort(axis=1)
print(d)  

# 배열의 최대값과 최소값 출력
print(np.max(d))  # 배열의 최대값 출력
print(np.min(d))  # 배열의 최소값 출력

# 배열 `a`의 타입 출력
print(type(a))  # 타입 출력