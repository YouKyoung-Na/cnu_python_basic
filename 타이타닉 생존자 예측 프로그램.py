#!/usr/bin/env python
# coding: utf-8

# ### 타이타닉 데이터를 활용한 이진 생존 분류
# - Download: https://www.kaggle.com/c/titanic

# ### 1. Data Description
# - total: 1309명
# - train.csv: 891x12
# - test.csv: 418x11
# 
# - Survive: 0=No, 1=Yes
# - pclass: 1=1st, 2=2nd, 3=3rd(숫자가 낮을수록 높은 등급)
# - SibSP = Sibilings-Spouses
# - Parch = Parents-Children

# ### pandas
# - pandas 처럼 2차원 공간에 온 것 = dataframe이라고 함
# - feature = 특징(열) = column
# - series = 특징 하나에 대한 data
# - table = 표
# - column = 세로(열)
# - row = 가로(행)

# ### 데이터(CSV,JSON)
# 
# - Pandas: 데이터 전처리, 탐색+분석
# - Scikit-learn: 머신러닝 (KNN etc..)
# - Tensorflow, PyTorch: 딥러닝

# ### 2. Import Module

# In[1]:


import pandas as pd  #데이터 전처리, 탐색, 분석
import numpy as np  #수치해석
import matplotlib.pyplot as plt  #그래프 그려주는 것


# ### 3. Data Load
# - 절대 경로 / 상대경로
#     - 웬만해선 상대경로 사용하기(다른 컴퓨터에서도 사용하기 위해)
# - .:현재  ..:뒤로가기(폴더 나옴)  /:들어가기(폴더 들어감)

# In[2]:


df_train = pd.read_csv('./dataset/train.csv')  #pd.read_csv() = pandas에 들어있는 csv파일을 알려주는 함수
df_test = pd.read_csv('./dataset/test.csv')
df_train.head()  #.head() = 처음부터 5건만 불러옴, .tail() = 끝에서부터 5건


# In[3]:


print(type(df_train))
print(type(df_test))


# ### 4. Data Exploration

# In[5]:


# Data Size 확인
print(df_train.shape)
print(df_test.shape)


# In[9]:


# Data Type 및 Null 확인
# Cabin, Age, Embarked: 3가지 feature는 Null 존재(결측치 존재)
# ->Null(결측치) 어떻게 처리할 것인지 고민
df_train.info()


# In[11]:


# Column 별 결측치(Missing Value) 합계
df_train.isnull().sum()


# In[14]:


#Feature 목록 추출 및 Type
print(df_train.columns)
print(df_train.columns.values) #list


# In[16]:


# 문자열(Object) 안나옴. Only 수치형 타입
df_train.describe()


# In[18]:


# count = unique => 다 다른 이름이다.(Name)
# Ticket은 중복되는 티켓이 여러 사람에게 동시 발행 => 가족 단위 추론
# => 티켓과 생존간의 상관관계(아마 의미 없을 듯)
df_train.describe(include=["O"])  #대문자 O


# In[19]:


#column category 확인
df_train["Survived"].unique()


# In[20]:


df_train["Survived"].value_counts()


# In[21]:


ratio = df_train["Survived"].value_counts()
labels=df_train["Survived"].unique()
plt.pie(ratio, labels=labels, autopct="%.1f%%")
plt.show()


# In[22]:


df_train['Pclass'].unique()


# In[23]:


df_train['Pclass'].value_counts()


# In[24]:


ratio = df_train["Pclass"].value_counts()
labels=df_train["Pclass"].unique()
plt.pie(ratio, labels=labels, autopct="%.1f%%")
plt.show()


# ### 5. 데이터 추출

# In[25]:


#Column 추출(1개)
df_train['Age']


# In[31]:


# Column 추출 여러개
df_train[['Age','Pclass','Name']]


# In[34]:


# iloc를 사용한 Row 추출 =>기존에 자동으로 생성된 index
df_train.iloc[3]

# loc를 사용한 Row 추출 => 우리가 생성한 label
#df_train.loc[label]


# ### 6. 데이터 분석

# In[35]:


# Pclass(티켓 등급), 티켓 등급별 생존율 분석
df_train[["Pclass", "Survived"]].groupby(["Pclass"]).mean()


# In[36]:


df_train[["Sex", "Survived"]].groupby(["Sex"]).mean()


# In[39]:


# 상관분석
# * -1~1의 값
# * 1에 가까울수록 양의 상관관계(니가 오르면 나도 오름)
# * -1에 가까울수록 음의 상관관계(니가 오르면 나는 내려감)
# * 0은 아무 관계도 없음
df_train.corr()


# In[40]:


# 히트맵 그래프
plt.matshow(df_train.corr())

