#!/usr/bin/python
#encoding=utf-8
import numpy as np
import pandas as pd
import time
import random
from math import sqrt,fabs,log
import sys
import sklearn
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
np.set_printoptions(threshold=100000)
X1=np.zeros((41,128))
X2=np.zeros(41)
Y1=np.zeros((97,128))
Y2=np.zeros(97)
department = ['城管局','工商联','公安','科技局','统战部','文广旅体局','自然资源局','住建水利局','政数局','发改局',
'法院','广发银行','行政服务中心','纪委','检察院','教育局','金融办','经促局','农商行','农业农村局',
'农业银行','气象局','人才办','人社局','社保局','生态环境局','市监局','数据中心','税务局','统计局',
'卫健局','消防大队','宣传部','应急管理局','镇街1','镇街2','镇街3','镇街4','镇街5','镇街6','镇街7']
tables= [[1,9,62],[1,7,12,15,38],[1,3,5,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,36,37,38,39,40,43,44,45,46,47,48,50,51,52,53,55,56,57,58,59,62,63,64,65,66,67,68,70,73,74,75,76,77,78,80,82,83,84,85,87,88,89,95],
[1,85],[1,61],[1],[1,3,15,25,38,40,47,58,68,97],[1,5,11,19,21,40,42,48,80,89,90,91,92,95,97],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96],
[1,3,42,49,60,61,71,94],[1,2,3,4,5,6,7,8,9,10,11,12,13,18,20,23,24,27,29,35,36,37,41,43,44,46,47,50,51,54,55,56,59,65,66,67,69,72,73,75,76,77,79,80,81,82,86,87,91,92,93,97],
[1,8,33,64,73],[1,2,3,4,5,7,8,9,11,13,16,17,18,19,20,21,22,23,24,25,26,32,36,41,42,43,44,45,46,48,51,53,56,59,62,69,79,81,82,86,89,90,97],
[1,2,3,5,7,8,9,10,11,13,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,35,36,39,40,41,43,44,45,46,47,49,52,54,55,56,57,58,60,61,62,63,66,67,68,69,70,72,73,74,75,76,77,79,82,83,84,87,88,92,96],
[1,3,5,7,8,9,10,11,12,13,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,38,39,42,43,44,45,46,47,49,50,51,52,55,57,58,59,60,62,63,64,65,67,68,70,71,74,75,76,77,78,81,82,83,84,85,86,87,88,89,93,94,95,96],
[1,4,10,76],[1,3,5,7,8,10,11,12,13,15,17,18,19,20,23,24,26,27,28,29,30,31,32,33,34,35,37,38,41,47,50,52,53,54,57,64,65,66,67,68,69,70,71,72,74,77,81,83,84,90,91,94,96],
[1,8,42],[1,3,12,13,37,39,40,41,42,49,60,61,79],[1,80],[1,3,12,28],[6,18,20,21,22,31,36,48,53,65,66,78],[1,2,7,9,10,13,15,25,30,34,35,37,39,43,46,48,54,63,74,75,78,79,80,83,85,86,87,93],
[1,12,17,34,45,49,56,60,70,72],[1,4],[1,5,17,26],[3,15,30,61],[15],[1,5,7,8,9,10,11,14,15,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,38,39,41,45,50,51,52,53,54,55,57,58,59,63,64,69,71,72,73,78,81,84,85,86,88,90,91,92,93,94,95,96],
[1],[40],[1,44,80,97],[28,71],[22],[2,4,6,14,16],[2,4,6,14,16],[1,2,4,6,14,16],[2,4,6,14,16],[2,4,6,14,16,25],[1,2,4,6,14,16],[2,4,6,14,16]]
t_name =['工商开业登记信息','流动人口表','纳税信用等级纳税人备案信息1','常住人口基本信息','变更登记信息','慈善组织','法定代表人正式信息表','用电量信息表','自然人信息表','房地产交易信息(JKJS1)',
'建设用地规划许可证信息','国税欠税信息','住房公积金个人公积金年度汇缴明细信息','民办非企业单位信息(JKMZ1)','高新技术企业认定信息','社会团体信息(JKMZ2)','企业用水信息(JKGZ1)','现场检查记录','建设工程规划许可证信息JKGH2','厂企信息(网格)',
'幼儿园','涉及职业危害作业场所','企业用户基本信息','土地交易信息表','人口基础信息表','用气信息','吊销登记信息','内资企业设立登记公告数据','财政直接支付信息(JKCZ2)','高企认定或拟认定名单',
'工贸企业安全生产标准化三级达标企业信息','发改局基建工程批准立项信息','对外贸易经营者备案登记情况','违法违章受处罚企业信息','年度奖励扶持企业情况','小作坊信息(网格)','律师基本信息采集表','组织机构代码信息','专业技术人员信息','统一社会信用',
'食品经营企业信息','环保信用严管企业','流动人口登记表','出租屋详细信息','残疾人基本信息','暂住证业务表','企业基本信息','高校','失信被执行人信息（法人或其他组织)','工程建设单位项目业绩信息(JKJS17)',
'佛山市安置残疾人单位信息(JKCL1)','水费收集信息(包含企业用水信息)','危险化学品生产许可证核发信息','单位房屋产权信息表','建设项目情况信息(JKJS3)','残疾人管理','房屋建筑工程竣工验收备案信息(JKJS6)','房地产开发预售明细信息(JKJS2)','商品房预售项目信息表','失信被执行人信息（自然人)',
'纳税信用等级纳税人备案信息2','建设工程规划管理验收合格证','外国文教类专家信息(JKRS1)','外商投资企业新设信息表','佛山市农业龙头企业信息','道路运输经营信息采集表','组织机构代码变更信息','重点建设项目计划信息','行政许可信息（住建）','企业拖欠水费信息收集',
'欠薪企业信息(JKRS3)','用电异常企业信息','外商实际投资情况','企业工法奖项信息','流动人口注销信息','居住证业务有效表','公有资产管理信息','个人职业技能信息','个人就业情况信息','名址库_房屋信息',
'行政处罚决定书','出租屋登记信息','企业专利信息','企业人力信息','金融机构信息','机动车信息表','个人基本信息','项目数据','危险品仓库数据表','建设用地规划许可证信息JKGH1',
'土地交易信息(JKGT1)','佛山市土地使用权信息(JKGT4)','外国人入境就业信息(JKRS6)','（市经信）广东省名牌产品','建设项目选址意见书信息','房地产开发企业信息表','名址库_门牌地址信息']
k_name = ['纳税','常住人口','雄鹰']
k_tables =[[3,12,61],[4],[35]]
#print(len(t_name))
rec_pri=[]
#for i in range(2):
    #print(np.array(tables[i]))
name = input("输入部门名称:")
#name = "税务局"
#keyword = input("输入关键词：")
keyword = "纳税"
if keyword in k_name:
    for i in range(len(k_tables[k_name.index(keyword)])):
        rec_pri.append(k_tables[k_name.index(keyword)][i])
    for i in range(len(rec_pri)):
        print("优先考虑",t_name[rec_pri[i]-1])
#print(rec[i])
key = department.index(name)
#key_ = t_name.index(keyword)
#key = 0 
key_ = 0
dis =15
m = np.zeros((41,97))
print(key)
print(key_)
for i in range(41):
    for j in range(len(tables[i])):
        m[i][tables[i][j]-1]=1
print(m)
class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths,item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
       # print ('Load user embeddings finished.')
        #self.X1={}
        for i in range(unum):
            X1[i] = 0.5*self.X[i][1]+0.5*self.X[i][0]
            #X1[i] = self.X[i][0]
            #a.append(list(X1.values())[i])
           # print(a)
           # print(X1[i])
        #print(list(X1.values())[1])
        #print(list(X1.values())[0])
        pca =PCA(n_components=2)
        pca_result = pca.fit_transform(X1)
        x = pca_result[:,0]
        y = pca_result[:,1]
        txt = ['1','2','3','4','5','6','7','8','9','10',
        '11','12','13','14','15','16','17','18','19','20',
        '21','22','23','24','25','26','27','28','29','30',
        '31','32','33','34','35','36','37','38','39','40','41']
        plt.scatter(x,y,color='r',marker='+')
        for i in range(len(x)):
            plt.annotate(txt[i],xy=(x[i],y[i]),xytext=(x[i]+0.00005,y[i]+0.00005))
       # plt.show()
        #print(0.5*self.X[0][1]+0.5*self.X[0][0])
        for i in range(unum):
            X2[i] = np.sum(np.abs((0.5*self.X[i][1]+0.5*self.X[i][0])-(0.5*self.X[key][1]+0.5*self.X[key][0])))
           # X2[i] = np.sum(np.abs((self.X[i][0])-(self.X[key][0])))
            #a.append(list(X1.values())[i])
           # print(a)
            #print(X2[i])

        from sklearn.metrics.pairwise import cosine_similarity
        array_X2 = np.array(cosine_similarity(X1)[key])
        print(array_X2)
        top_k = 10
        top_k_index = array_X2.argsort()[::-1][1:top_k+1]
        print(top_k_index)
        array =np.zeros(97)
        for i in range(len(tables[key])):
            array[tables[key][i]-1]= 1
        print(array)
        for i in range(top_k):
            print(department[top_k_index[i]])
        
        for j in range(97):
            b = j+1
            if b in tables[key]:
                continue
            for i in range(top_k):
                array[j] += array_X2[top_k_index[i]]*m[top_k_index[i]][j]
        print(array)
        d = array.argsort()[::-1][0:dis]
        print(d)
        #print(Counter(array))
       # d = sorted(Counter(array).items(),key=lambda x: x[1], reverse = True)
       # nn=[]
       # for x in d:
           # nn.append(x[0])
        #print(nn)
        #print(len(nn))
       # print(t_name[0])
        rec = []
        for i in range(len(d)):
            rec.append(t_name[d[i]])
        for i in range(len(d)):
            print(rec[i])
        #print(rec)
        #print(d)
        #print(np.unique(array,return_counts=True))
        #print(array[1])
        #a
        #if(top_k_index)
        #print(cosine_similarity(X1))
        #print(np.sum(np.abs((0.5*self.X[2][1]+0.5*self.X[2][0])-(0.5*self.X[1][1]+0.5*self.X[1][0]))))
        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        #print ('Load item embeddings finished.')
        #print(0.5*self.Y[42][1]+0.5*self.Y[42][0])
        for j in range(inum):
            Y1[j] = 0.5*self.Y[j][1]+0.5*self.Y[j][0]
            #X1[i] = self.X[i][0]
            #a.append(list(X1.values())[i])
           # print(a)
           # print(X1[i])
        #print(list(X1.values())[1])
        #print(list(X1.values())[0])
        pca =PCA(n_components=2)
        pca_result = pca.fit_transform(Y1)
        x1 = pca_result[:,0]
        y1 = pca_result[:,1]
        txt1 = ['1','2','3','4','5','6','7','8','9','10',
        '11','12','13','14','15','16','17','18','19','20',
        '21','22','23','24','25','26','27','28','29','30',
        '31','32','33','34','35','36','37','38','39','40',
        '41','42','43','44','45','46','47','48','49','50',
        '51','52','53','54','55','56','57','58','59','60',
        '61','62','63','64','65','66','67','68','69','70',
        '71','72','73','74','75','76','77','78','79','80',
        '81','82','83','84','85','86','87','88','89','90',
        '91','92','93','94','95','96','97']
        plt.scatter(x1,y1,color='r',marker='+')
        for j in range(len(x1)):
            plt.annotate(txt1[j],xy=(x1[j],y1[j]),xytext=(x1[j]+0.00005,y1[j]+0.00005))
        #plt.show()
        #print(0.5*self.X[0][1]+0.5*self.X[0][0])
        for j in range(inum):
            Y2[j] = np.sum(np.abs((0.5*self.Y[j][1]+0.5*self.Y[j][0])-(0.5*self.Y[key][1]+0.5*self.Y[key][0])))
           # X2[i] = np.sum(np.abs((self.X[i][0])-(self.X[key][0])))
            #a.append(list(X1.values())[i])
           # print(a)
            #print(X2[i])
        from sklearn.metrics.pairwise import cosine_similarity
        #print(cosine_similarity(Y1))
        array_Y2 = np.zeros(97)
        
        print(array_Y2)
        top_k = 10
        top_k_index = []

        arrays = np.zeros(97)
        #for i in range(top_k):
            #print(t_name[top_k_index[i]])
        for i in range(len(tables[key])):
            arrays[tables[key][i]-1]= 1
        print(arrays)
        #for i in range(top_k):
           # print(department[top_k_index[i]])
        
        for j in range(97):
            array_Y2 = np.array(cosine_similarity(Y1)[j])
            top_k_index = array_Y2.argsort()[::-1][1:top_k+1]
            #print(top_k_index)
            #print(array_Y2)
            b = j+1
            if b in tables[key]:
                continue
            for i in range(top_k):
                arrays[j] += array_Y2[top_k_index[i]]*m[key][top_k_index[i]]
                #arrays[j] += m[j][top_k_index[i]]*array_Y2[top_k_index[i]]
        print(arrays)
        d1 = arrays.argsort()[::-1][0:dis]
        print(d1)
        rec1 = []
        for i in range(len(d1)):
            rec1.append(t_name[d1[i]])
        for i in range(len(d1)):
            print(rec1[i])
        #for i in range(top_k):
           # arrays += tables[top_k_index[i]]
        #print(np.array(array))
        #print(Counter(array))
        #d = sorted(Counter(arrays).items(),key=lambda x: x[1], reverse = True)
       # nns=[]
       # for x in d:
          #  nn.append(x[0])
        #print(nn)
        #print(len(nn))
       # print(t_name[0])
       # recs = []
       # for i in range(len(nns)):
           # recs.append(t_name[nn[i]-1])
       # for i in range(10):
          #  print(rec[i])
        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        #print(self.R)
       # print(self.T)
        #print(self.ba)
       # print ('Load rating finished.')
        #print ('train size : ', len(self.R))
        #print ('test size : ', len(self.T))

       # self.initialize();
       # self.recommend();

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []
    
        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            #print sourcefile
            with open(sourcefile) as infile:
                
                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
               # print ('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user)-1, int(item)-1, int(rating)])
                ba += int(rating)
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user)-1, int(item)-1, int(rating)])

        return R_train, R_test, ba



if __name__ == "__main__":
    unum = 41
    inum = 97
    ratedim = 10
    userdim = 10
    itemdim = 10
    train_rate = 1.0#float(sys.argv[1])

    user_metapaths = ['ubu', 'ubcabu']
    item_metapaths = ['bub', 'bcab']
    #user_metapaths = ['ubu']
    #item_metapaths = ['bub']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'
    #user_metapaths = ['ubu_' + str(train_rate) + '.embedding', 'ubcibu_'+str(train_rate)+'.embedding', 'ubcabu_'+str(train_rate)+'.embedding']
    
    #item_metapaths = ['bub_'+str(train_rate)+'.embedding', 'bcib.embedding', 'bcab.embedding']
    trainfile = '../data/ub_'+str(train_rate)+'.train'
    testfile = '../data/ub_'+str(train_rate)+'.test'
    steps = 100
    delta = 0.02
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.01
    reg_u = 1.0
    reg_v = 1.0
    print ('train_rate: ', train_rate)
    print ('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print ('max_steps: ', steps)
    print ('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b', beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    HNERec(unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
   # print(self.X)