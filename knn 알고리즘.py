import pandas as pd
import numpy as np
import matplotlib as plt
from operator import itemgetter

##데이터 불러오기
glass = pd.read_csv("C:\\Users\\Administrator\\Desktop\\PythonCode\\glass.csv")
glass = glass.drop('number',1)
glass = round(glass,3)
glass

##데이터를 섞은 후 트레이닝 데이터셋과 테스트 데이터셋으로 나누기
glass_train = glass.sample(frac=0.5)
glass_test = glass.loc[~glass.index.isin(glass_train.index)]
glass_train
glass_test

##트레이닝 데이터셋과 테스트 데이터셋에서 라벨과 데이터를 구분
glass_train_data = glass_train.loc[:,'RI':'Fe']
glass_train_label = glass_train.loc[:,'label']
glass_train_data
glass_train_label

glass_test_data = glass_test.loc[:,'RI':'Fe']
glass_test_label = glass_test.loc[:,'label']
glass_test_data
glass_test_label

##리스트화 시키기
glass_test_data_list = glass_test_data.values.tolist()
glass_train_data_list = glass_train_data.values.tolist()
glass_test_label_list = glass_test_label.values.tolist()
glass_train_label_list = glass_train_label.values.tolist()
glass_test_label_list


def knn(k,train_data, test_data, train_label):
    dist = 0
    result = []
    ##test data 와 train data 연산(연산횟수 : test_data개수 * train_data개수)
    for test in test_data:
        dist_list = []
        vote_label_list = []
        cnt  = 0
        for train in train_data:
            ##유클리드 거리 구하기
            a = np.array(test)
            b = np.array(train)
            dist = plt.mlab.dist(a,b)
            ##거리값을 저장, 거리값 저장시 [거리값, 라벨]형태의 size 2의 list로 저장
            dist_list.append([dist,train_label[cnt]])
            cnt += 1
            ##거리값들을 가까운순으로 정렬
            dist_list = sorted(dist_list, key=itemgetter(0))
            ##k의 개수만큼 추출
            dist_list_k = dist_list[:k]
            ##투표를 위한 label값 추출
        vote_label_list = np.array(dist_list_k)[:,1]
        print(vote_label_list)
            
            
        
            
knn(7,glass_train_data_list,glass_test_data_list, glass_train_label_list)



