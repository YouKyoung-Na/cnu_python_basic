#!/usr/bin/env python
# coding: utf-8

# - 구매자로부터 가장 사고싶은 과일 사진을 보내면
# 
# - 구매자가 가장 많이 요청하는 과일을 판매 목록 선정
# 
# - 사람들이 과일 사진을 너무 많이 보내줬는데, 이걸 하나하나 무슨 과일인지 체크할 사람이 없음
# 
# > 데이터: 사람들이 보낸 과일 사진, 정답은 따로 X
# 
# > 정답 X => 분류 모델 사용할 수 없음
# 
# > 클러스터링(군집) => 비지도학습(정답 X)  -> 보험사에서 클러스터링 모델 많이 사용
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# ### 1. 데이터 로드

# In[3]:


fruits = np.load('fruits_300.npy')


# In[5]:


# 총 300장 사진, 사진 1장 당 100*100 픽셀
print(fruits.shape)


# In[6]:


# 흑백사진 0~255의 정수값
print(fruits[0,0,:])


# In[9]:


# 사과사진
# 숫자가 0에 가까울수록 검게 나타남
plt.imshow(fruits[0], cmap='gray')
plt.show()


# In[11]:


plt.imshow(fruits[0], cmap='gray_r')  #r=reverse, 반전시킴
plt.show()


# In[13]:


#파인애플, 바나나
fig, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')

