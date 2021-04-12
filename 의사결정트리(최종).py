import pandas as pd
import math as mt
from collections import Counter

##데이터 불러오기
#data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\PythonCode\\zoo.csv',encoding = 'CP949')
#label = 'type' #라벨에 해당하는 컬럼의 이름을 입력해야함
#data = data.iloc[:,1:]

data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\PythonCode\\fatliver2.csv',encoding = 'CP949')
label = 'FATLIVER'

## column_value의 입력값은 pandas dataframe으로 받은 후, 컬럼을 list화 시켜서 입력한다. Ex) column = dataframe[:,'column_name']
## column_value 함수는  dataframe의 해당 컬럼의 값을 리스트화 한 것과, 해당 컬럼값의 종류를 리턴한다.
## Ex) [1,3,2,3,2,1,3,3,3,1,2,3 ...., 3,2,1] [1,2,3]
def column_value(column):
    column_list = []    
    for i in column:
        column_list.append(i)
    column_type = list(set(column_list))
    return column_list, column_type # 해당 컬럼의 값, 해당 컬럼 값의 종류

## 분할전 엔트로피값 계산하기
## 해당 컬럼에 대한 엔트로피 값을 계산한다. 
## Ex) [T,T,F,T,F,T,F,T,T,T]
##     전체 : 10  T의 개수 : 7   F의 개수 : 3
## 분할전 엔트로피 = 7/10 * log(7/10, 2) + 3/10 * log(3/10, 2)
def column_entropy(column):
    label_list = column_value(column)[0]
    label_type = column_value(column)[1]
        
    column_entropy = 0
    #해당 컬럼 값의 종류를 i에 입력 Ex) [1,2,3]
    for i in label_type:
        rate_value = 0     
        cnt = 0
        rate = 0
        #해당 컬럼 list에서 값의 종류별 개수 세기
        for j in label_list:
            if j == i:
               cnt += 1
        rate = cnt/len(label_list)
        rate_value = -rate * mt.log(rate,2)
        column_entropy += rate_value
    return column_entropy

## 분할 후 엔트로피값 계산하기 
## 해당 컬럼에 대하여 라벨값의 엔트로피를 측정하는 함수
## Ex)
## label_list = [T,T,F,T,F,T,F,T,....,T,F,T]
## column_list = [1,2,3,2,1,3,1,....,1,3,2], column_list값의 종류 = [1,2,3]
## column_value "1" = [T,F,T,T,F,T] --> 엔트로피 계산(column_entropy 함수)
## column_value "2" = [F,T,T,T,F,T] --> 엔트로피 계산(column_entropy 함수)
## column_value "3" = [T,F,F,T,F,T] --> 엔트로피 계산(column_entropy 함수)
## 분할 후 엔트로피 = (column_value"1"의 개수/column_list 개수) * column_value"1" 엔트로피 +
##                 (column_value"2"의 개수/column_list 개수) * column_value"2" 엔트로피 +
##                 (column_value"3"의 개수/column_list 개수) * column_value"3" 엔트로피 
def after_divide_entropy(column, label):
    label_list = column_value(label)[0]
    column_list = column_value(column)[0]
    column_type = column_value(column)[1]
    after_divide_entropy = 0
    #해당 column값의 종류를 i 에 입력 
    for i in column_type:
        value_list = []
        #i값과 일치하는 column_list의 index를 label_list에 대입하여 해당 값을 value_list에 추가
        for j, k in enumerate(column_list):
            if i == k:
                value_list.append(label_list[j])
        after_divide_entropy += len(value_list)/len(column_list) * column_entropy(value_list)
    return after_divide_entropy

###########################################################################################

##해당 data의 label에 대한 분할 전 엔트로피와  column별 label에 대한 분할 후 엔트로피 출력
print('분할 전 엔트로피 :', round(column_entropy(data.loc[:,label]),3))
print('====================================================')

data_column_list = list(data.columns)
data_column_list.remove(label)
data_column_list

##분할 후 엔트로피가 가장 작은 값이 정보획득량이 가장 높다(의사결정트리 node 생성의 기준)
for i in data_column_list:
    print( i,'분할 후 엔트로피 :', round(after_divide_entropy(data.loc[:,i],data.loc[:,label]),3) )

print('====================================================')

for i in data_column_list:
    print( i,'의 정보획득량 :', round(column_entropy(data.loc[:,label]) - after_divide_entropy(data.loc[:,i],data.loc[:,label]),3) )

############################################################################################
    
## 해당 dataframe에서 label에 대한 분할 후 엔트로피가 가장 작은 컬럼을 리턴
def min_entropy_column(data,label):
    from operator import itemgetter
    data = data
    label = label
    entropy_list = []
    #data의 컬럼 네임값들을 리스트로 저장하고 label 컬럼의 컬럼네임 제거
    data_column_list = list(data.columns)
    data_column_list.remove(label)
    #data의 컬럼 네임값 별 분할 후 엔트로피 계산 
    for i in data_column_list:
        entropy_list.append([after_divide_entropy(data.loc[:,i],data.loc[:,label]), i])
    return min(entropy_list,key=itemgetter(0))[1]



##의사결정나무를 만드는 함수
def tree(data,label,maxdepth):
    data = data #의사결정 나무를 그릴 data
    label = label #라벨컬럼 인식
    maxdepth = int(maxdepth) #의사결정나무의 최대 깊이를 지정
    tree_dict = {} 
    # 의사결정나무는  dictionary 형태로 저장, key 값은 의사결정나무의 노드이며, value는 해당 노드의 dataframe을 저장 
    # depth1은 최초 사전에 저장하는 과정이기 때문에 따로 처리한다.
    min_column = min_entropy_column(data, label) # data에서 엔트로피값이 가장 최소인 컬럼 지정
                                                 # Ex) AGE 컬럼인 것으로 가정
    vis_inform = [] # 시각화를 하기 위해 필요한 정보 저장
    for i in set(data[min_column]): # 엔트로피값이 가장 최소인 컬럼의 값의 종류를 i에 대입
                                    # Ex) AGE 컬럼의 값의 종류는 '20대', '30대', '40대'로 가정
        key = str(min_column) + str(' ') + str(i) #key의 이름은 '해당컬럼이름 + 값'으로 지정된다. Ex) column이름 : AGE / 값 : 20대 ---> 'AGE 20대'
        value = data[data[min_column] == i] # 엔트로피가 최소인 컬럼의 값들 중에 i와 같은 값을 가지는 dataframe을 value로 저장
        tree_dict[key] = value # 사전에 추가
        
        value_inform = Counter(value.loc[:,label]).most_common(1)[0] # dataframe 형태인 value 변수에서 label 컬럼의 값들중 가장 빈도수가 높은 값
        vis_inform.append([key,len(tree_dict[key]),value_inform[0],value_inform[1]]) #시각화를 위한 vis_inform list에 값 추가
                        # key값, key값인 value의 행의 개수(해당 key값의 건수), label 컬럼의 값등 중 가장 빈도수가 높은 값, 해당 값의 빈도
    # Ex) 위의 과정을 실행하면 tree_dict의 key 값은 'AGE 20대' 'AGE 30대' 'AGE 40대', 각각에 dataframe형태로 value가 들어가있음
    # depth1은 생성을 했기 때문에 maxdepth-1번 만큼 반복
    for m in range(maxdepth-1):
        tree_dict_key_list = list(tree_dict.keys()) #tree_dict 사전의 key값들의 목록(노드를 분할하기 위해) 
                                                    #Ex) key 값 리스트에 'AGE 20대' 'AGE 30대' 'AGE 40대'
        for i in tree_dict_key_list: # Ex) i = 'AGE 20대'
            data_frame = tree_dict[i] # Ex) tree_dict에서 key값이 'AGE 20대'인 값을 data_frame에 저장
            min_column = min_entropy_column(data_frame,label) # data_frame에서 엔트로피 값이 가장 낮은 컬럼 선정
                                                              # EX) AGE 20대인 data_frame에서 최소 엔트로피 컬럼이 SMOKE로 가정
            # 의사결정트리가 분할이 될때 기존에 분할된 컬럼으로 분할이 되면 안돼기 때문에 분할 기록을 저장하는 곳(분할이 어떻게 진행되었는지는 key값이 설명해줌)
            key_element = [] #Ex) key값 'AGE 20대 - SMOKE 흡연 - GENDER 여자 - DRINK 많음' ------> key_element = [AGE, SMOKE, GENDER, DRINK] 
            for k in i.split(' - '):
                key_element.append(k.split(' ')[0])
                
            if len(set(data_frame[label])) != 1 and min_column not in key_element: #분할이 되지 않는 조건 2가지 = dataframe의 label 값의 종류가 1이거나(분류 완료), 기존에 선정한 컬럼으로 또 분할을 하려고 할때
                for j in set(data_frame[min_column]):                              #SMOKE컬럼의 값은 '흡연','금연'으로 가정하면, j = '흡연','금연'
                    key = str(i) + str(' - ') + str(min_column) + str(' ') + str(j)  #key값은 분할이 진행될때 기존 이름에 덧붙여서 이름을 또 생성한다. 이를통해 분할 진행과정을 보여줄 수 있다.
                                                                                     #Ex) 'AGE 20대' key 값이 SMOKE로 분할된다면 key 이름은 'AGE 20대 - SMOKE 흡연','AGE 20대 - SMOKE 금연'으로 사전에 새로 추가한다.
                    value = data_frame[data_frame[min_column] == j] # 각각 SMOKE가 '흡연'과 '금연'에 해당하는 data_frame 생성
                    tree_dict[key] = value # 사전에 추가
                    value_inform = Counter(value.loc[:,label]).most_common(1)[0]
                    vis_inform.append([key,len(tree_dict[key]),value_inform[0],value_inform[1]])
                tree_dict.pop(i) # 기존 컬럼은 분할을 마쳤기 때문에 삭제한다
                                 # Ex) 'AGE 20대 - SMOKE 흡연','AGE 20대 - SMOKE 금연'이 생성되었으므로 'AGE 20대'인 key값과 value값을 삭제
                                 # 위의 과정을 'AGE 30대' 'AGE 40대'에도 반복
                                 # vis_inform에서는 key값을 삭제하지 않기 때문에 트리가 생성되는 누적 과정을 전부 보여준다.
    
    return tree_dict, sorted(vis_inform) # 의사결정트리 생성이 완료되면 tree_dict 사전과, 시각화를 위한 list형태의  vis_inform 반환

tree_dict = tree(data,label,4)[0]
vis_inform = tree(data,label,4)[1]
############################### 시각화 ##########################################################

#위의 함수에서 maxdepth를 지정해주긴 하였으나, maxdepth까지 분류가 안될 수 있기 때문에 분할된 트리의 maxdepth를 구한다
max_depth = []
for i in tree_dict.keys(): # key값이 분할 과정을 보여주고 있기 때문에 maxdepth를 알려줄수 있다
    max_depth.append(len(i.split(' - ')))
max_depth = max(max_depth)
max_depth

#시각화에서 노드를 만들 때 중복된 노드를 생성하면 안돼기 때문에  생성된 노드와 생성되지 않은 노드를 구분할 수 있는 list 생성
#이중 리스트로 구현이 되어 있는데 리스트 안의 첫번째 리스트는 depth1, 두번째 리스트는 depth2....이런식으로 maxdepth만큼 생성
tree_depth_list = []
for i in range(max_depth):
    tree_depth_list.append(['★'])

## 최종 라벨 타입에 따라 컬러 변경하는 코드
color_list = ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2"]
node_color = {} #컬러값 저장을 위한 딕셔너리
for i,j in enumerate(set(data[label])):
    key = j
    value = color_list[i]
    node_color[key] = value     

#Graphviz에 적용가능한 컬러맵(아래 사이트의 Brewer color 식별코드를 알수 있음 https://gist.github.com/rwl/9996408
#Graphviz 색상표 https://www.graphviz.org/doc/info/colors.html#brewer_license
#Graphviz node 모양 https://www.graphviz.org/doc/info/shapes.html

from graphviz import Digraph  #그림그리는 툴로 graphviz 추가 
dot=Digraph(comment='TEST', format='svg') #Digraph로 그래프를 그릴 변수 지정 
dot.node('basic', 'Data'+ str('\n') + \
         str('총건수:') + str(len(data)) + str('건'), color='black', fillcolor='khaki1', shape='folder') #최상위 노드 생성 (노드명, 노출될 라벨, 모양)

for i in vis_inform: # vis_inform = [[key값, key값인 value의 행의 개수(해당 key값의 건수), label 컬럼의 값등 중 가장 빈도수가 높은 값, 해당 값의 빈도],...]
    node_name = i[0] # 노드명은 key값이 된다  Ex) 노드명이 'AGE 20대 - SMOKE 흡연' 이라 가정
    split = i[0].split(' - ') # Ex) split = ['AGE 20대', 'SMOKE 흡연']
    for j in range(max_depth): # vis_inform의 list 하나 당 maxdepth 만큼 반복
        ## 의사결정 트리에서 같은 depth에서, 해당 노드명은 옆에 있는 노드명과 무조건 다르다
        ## Ex) AGE 20대 - SMOKE 흡연, AGE 20대 - SMOKE 금연, AGE 30대 - SMOKE 흡연, AGE 30대 - SMOKE 금연, AGE 40대
        ##     --> depth2만 보면 --> SMOKE 흡연, SMOKE 금연, SMOKE 흡연, SMOKE 금연, (노드없음) 
        ##     따라서 노드를 생성할때마다 tree_depth_list에 split의 요소를 depth에 맞게 추가시켜 주고 해당 depth의 마지막 값들하고만 비교하면 중복을 막을 수 있다
        if j == 0 and tree_depth_list[0][-1] != split[0]: #depth1은 basic과 연결되기 때문에 따로 연산한다., tree_depth_list에서 depth1에 해당하는 tree_depth_list[0]의 마지막 값이 split의 같은 depth의 이름과 다르면 노드를 생성한다.
            if node_name in tree_dict.keys(): # tree_dict.keys()는 분할이 가장 마지막에 된 key값들이다. 분할이 가장 마지막에 된 노드들에만 결과와 정확도를 출력한다
                dot.node(node_name,str(split[0]) + str('\n') + \
                                   str('해당건수:') + str(i[1]) + str('건(') + str( round(i[1]/len(data)*100,2) ) + str('%') + str(')') + str('\n') + \
                                   str('결과:') + str(label) + str(' ') + str(i[2]) + str('\n') + \
                                   str('정확도:') + str(i[3]) + str('건') + str('(') + str(round(i[3]/i[1]*100,2)) + str('%') + str(')') + str('\n') \
                                   ,fillcolor=node_color[i[2]],shape = 'note', color='grey30')
            else: #가장 마지막에 분할이 되지 않은 key값이면 노드만 생성
                dot.node(node_name, split[j], fillcolor = 'grey76', color = 'grey30', shape = 'oval', fontcolor='white')
            dot.edge('basic',node_name)
            tree_depth_list[0].append(split[0])
        else: #j가 0이 아닐때
            try: # try가 존재하는 것은 maxdepth만큼 분할되지 않은 경우 split[j]를 불러오면 오류가 난다. 따라서 이 경우, tree_depth_list에 분할이 되지 않았다는 것을 알려주기위해 ' ' 를 추가
                if tree_depth_list[j][-1] != split[j]: # 해당 깊이의 바로 옆에 노드와 노드명이 다르면 노드 생성(위와 같음)
                    if node_name in tree_dict.keys():
                        dot.node(node_name,str(split[j]) + str('\n') + \
                                           str('해당건수:') + str(i[1]) + str('건(') + str( round(i[1]/len(data)*100,2) ) + str('%') + str(')') + str('\n') + \
                                           str('결과:') + str(label) + str(' ') + str(i[2]) + str('\n') + \
                                           str('정확도:') + str(i[3]) + str('건') + str('(') + str(round(i[3]/i[1]*100,2)) + str('%') + str(')') + str('\n') \
                                           ,fillcolor = node_color[i[2]],  shape = 'note', color='grey30')
                                 
                    else:
                        dot.node(node_name,split[j],fillcolor = 'grey76', color = 'grey30', shape = 'oval', fontcolor='white')
                    ##선을 연결하기 위해선 상위 노드를 인식해야 한다. Ex) 'AGE 20대, SMOKE 흡연'의 상위노드 --> 'AGE 20대'
                    upper_node = ''
                    for k in node_name.split(' - ')[:-1]: 
                        upper_node += str(k) + str(' - ')
                    upper_node = upper_node[:-3]
                    
                    dot.edge(upper_node,node_name)
                    tree_depth_list[j].append(split[j])
            except:
                 tree_depth_list[j].append(' ') # ' '이 추가되면 같은 깊이의 옆에 노드가 무조건 생기게 된다.                    



## node 컬러 설명 페이지 https://www.graphviz.org/doc/info/colors.html
## node 모양 설명 페이지 https://www.graphviz.org/doc/info/shapes.html
######### 그래프의 각 그래픽적 요소, 즉 스타일을 선택 ####################
styles = {
    'graph': {
        'label': 'Decision Graph',
        'fontsize': '16',
        'fontcolor': 'black',
        'bgcolor': '#f7f9f9',
        'rankdir': 'TB',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'fontsize' : '12',
        'fontcolor': 'black',
        'style': 'filled',
    },
    'edges': {
        'style': 'none',
        'color': 'grey30',
        'arrowhead': 'close',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'white',
    }
}
              

################ 앞에서 정한 스타일 적용 ##############################
def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph


print(dot.source) #dot에 들어가 있는 정보 확인
apply_styles(dot, styles) #스타일 적용 → 콘솔창에 그래프 그려짐
dot.render('C:\\Users\\user\\Documents\\test.gv', view=True) #파일 저장 및 새 창으로 그래프 출력