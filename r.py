import pandas as pd
import numpy as np
import random
import os
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
random.seed(100)

#load dataset
#user_keywords = pd.read_csv("user_keywords.csv")
user_keywords = pd.read_csv("u.csv")
'''
k = ['12345','热线','A级纳税人','艾滋病自愿免费咨询检测点','变更登记信息','补贴','不动产','测试特定部门','测试条件共享','产业',
'常住人口','常住人口基本信息','出租屋','从业人员体检登记表','村级工业园','工业园','地名','地图','电动车','发改','法人','房屋'
,'房屋信息','房屋信息表'
,'妇联微信_工作人员'
,'高企'
,'高新技术企业'
,'各级政府质量奖'
,'工商'
,'工商开业'
,'工商开业登记'
,'工商开业登记信息'
,'开业'
,'开业登记'
,'公积金'
,'公交'
,'公交GPS数据信息'
,'公交站亭数据表'
,'广东省名牌产品'
,'广亚铝业有限公司'
,'行政处罚'
,'行政处罚信息'
,'行政处罚信息（地税）'
,'行政审批事项'
,'行政许可'
,'行政许可信息'
,'合同基本资料'
,'黑名单'
,'黑烟车'
,'互联网'
,'户籍'
,'户联系'
,'婚姻'
,'婚姻案卷信息'
,'机动车'
,'机动车信息表'
,'检察院'
,'建筑物'
,'建筑物信息点'
,'紧缺适用人才申报信息'
,'空气质量'
,'劳模'
,'流动人口'
,'流动人口表'
,'律师事务所基本信息采集表'
,'门牌'
,'门诊诊疗信息'
,'名址库'
,'纳税'
,'纳税大户'
,'纳税大户光荣榜'
,'纳税人纳税信用等级信息查询'
,'纳税人税收收入情况信息'
,'纳税信用等级纳税人备案信息'
,'企业'
,'企业所得税'
,'所得税'
,'千人计划'
,'人才'
,'欠薪'
,'人口'
,'人口基础'
,'人口基础信息'
,'软件企业'
,'商标'
,'上市'
,'设备'
,'社保'
,'食品经营企业信息'
,'食品流通企业'
,'矢量'
,'市场主体'
,'市场主体登记信息'
,'市民卡'
,'事件上报'
,'水'
,'水库'
,'死亡'
,'四上'
,'四上企业'
,'土地'
,'外资企业设立公告数据'
,'网点信息'
,'网格'
,'网格化'
,'网格化_巡查发现问题单表'
,'网格信息'
,'污染源自动监测统计数据表'
,'无纸化'
,'物流'
,'物业'
,'先进制造业'
,'项目管理'
,'销售'
,'信访表'
,'信用'
,'雄鹰'
,'学校'
,'学校教师信息'
,'巡查'
,'巡查工单'
,'巡查记录'
,'巡查问题'
,'遥感'
,'药店'
,'一般纳税人应纳税额表'
,'医保'
,'医院'
,'用水'
,'幼儿园'
,'在校学生'
,'在校学生信息'
,'增值税'
,'政府机构'
,'政务地图'
,'政务地图遥感影像底图'
,'政务管理'
,'执委会成员名单'
,'治安及交通管理严重失信名单'
,'住房公积金个人公积金年度汇缴明细信息'
,'专利'
,'专业技术人员信息']
#print(k[141])
'''
k = ['工商开业登记信息','流动人口表','纳税信用等级纳税人备案信息1','常住人口基本信息','变更登记信息','慈善组织','法定代表人正式信息表','用电量信息表','自然人信息表','房地产交易信息(JKJS1)',
'建设用地规划许可证信息','国税欠税信息','住房公积金个人公积金年度汇缴明细信息','民办非企业单位信息(JKMZ1)','高新技术企业认定信息','社会团体信息(JKMZ2)','企业用水信息(JKGZ1)','现场检查记录','建设工程规划许可证信息JKGH2','厂企信息(网格)',
'幼儿园','涉及职业危害作业场所','企业用户基本信息','土地交易信息表','人口基础信息表','用气信息','吊销登记信息','内资企业设立登记公告数据','财政直接支付信息(JKCZ2)','高企认定或拟认定名单',
'工贸企业安全生产标准化三级达标企业信息','发改局基建工程批准立项信息','对外贸易经营者备案登记情况','违法违章受处罚企业信息','年度奖励扶持企业情况','小作坊信息(网格)','律师基本信息采集表','组织机构代码信息','专业技术人员信息','统一社会信用',
'食品经营企业信息','环保信用严管企业','流动人口登记表','出租屋详细信息','残疾人基本信息','暂住证业务表','企业基本信息','高校','失信被执行人信息（法人或其他组织)','工程建设单位项目业绩信息(JKJS17)',
'佛山市安置残疾人单位信息(JKCL1)','水费收集信息(包含企业用水信息)','危险化学品生产许可证核发信息','单位房屋产权信息表','建设项目情况信息(JKJS3)','残疾人管理','房屋建筑工程竣工验收备案信息(JKJS6)','房地产开发预售明细信息(JKJS2)','商品房预售项目信息表','失信被执行人信息（自然人)',
'纳税信用等级纳税人备案信息2','建设工程规划管理验收合格证','外国文教类专家信息(JKRS1)','外商投资企业新设信息表','佛山市农业龙头企业信息','道路运输经营信息采集表','组织机构代码变更信息','重点建设项目计划信息','行政许可信息（住建）','企业拖欠水费信息收集',
'欠薪企业信息(JKRS3)','用电异常企业信息','外商实际投资情况','企业工法奖项信息','流动人口注销信息','居住证业务有效表','公有资产管理信息','个人职业技能信息','个人就业情况信息','名址库_房屋信息',
'行政处罚决定书','出租屋登记信息','企业专利信息','企业人力信息','金融机构信息','机动车信息表','个人基本信息','项目数据','危险品仓库数据表','建设用地规划许可证信息JKGH1',
'土地交易信息(JKGT1)','佛山市土地使用权信息(JKGT4)','外国人入境就业信息(JKRS6)','（市经信）广东省名牌产品','建设项目选址意见书信息','房地产开发企业信息表','名址库_门牌地址信息']
"""
   user_id                                   keywords
0      113  新闻推荐|资讯推荐|内容推荐|文本分类|人工分类|自然语言处理|聚类|分类|冷启动
1      143                         网络|睡眠|精神衰弱|声音|人工分类
2      123                          新年愿望|梦想|2018|辞旧迎新
3      234                      父母|肩头|饺子|蔬菜块|青春叛逆期|声音
4      117       新闻推荐|内容推荐|文本分类|人工分类|自然语言处理|聚类|分类|冷启动
5      119            新闻推荐|资讯推荐|人工分类|自然语言处理|聚类|分类|冷启动
6       12              新闻推荐|资讯推荐|内容推荐|文本分类|聚类|分类|冷启动
7      122                   机器学习|新闻推荐|梦想|人工分类|自然语言处理
"""
keyword_list = [] 
def fun():
    os.system("python .py")
def date_process(user_item):
    """user_item is a DataFrame, column=[user_id, keywords]   
    1. user_item: user and item information, user_id, keywords, keyword_index
    2. user_index: user to index
    3. item_index：item to index
    """
    user_item["keywords"] = user_item["keywords"].apply(lambda x: x.split("、"))
    
    for i in user_item["keywords"]:
        keyword_list.extend(i)
        
    #word count
    item_count = pd.DataFrame(pd.Series(keyword_list).value_counts()) 
    # add index to word_count
    item_count['id'] = list(range(0, len(item_count)))
    
    #将word的id对应起来
    map_index = lambda x: list(item_count['id'][x])
    user_item['keyword_index'] = user_item['keywords'].apply(map_index) #速度太慢
    #create user_index, item_index
    user_index = { v:k for k,v in user_item["user_id"].to_dict().items()}
    item_index = item_count["id"].to_dict()
    return user_item, user_index, item_index

user_keywords, user_index, keyword_index = date_process(user_keywords)

def create_pairs(user_keywords, user_index):
    """
    generate user, keyword pair list
    """
    pairs = []
    def doc2tag(pairs, x):
        for index in x["keyword_index"]:
            pairs.append((user_index[x["user_id"]], index))
    user_keywords.apply(lambda x: doc2tag(pairs, x), axis=1) #速度太慢
    return pairs

pairs = create_pairs(user_keywords, user_index)
print(user_keywords.iloc[7])

def build_embedding_model(embedding_size = 128, classification = False):
    """Model to embed users and keywords using the Keras functional API.
       Trained to discern if a keyword is clicked by user"""
    
    # Both inputs are 1-dimensional
    user = Input(name = 'user', shape = [1])
    keyword = Input(name = 'keyword', shape = [1])
    
    # Embedding the user default: (shape will be (None, 1, 50))
    user_embedding = Embedding(name = 'user_embedding',
                               input_dim = len(user_index),
                               output_dim = embedding_size)(user)
    
    # Embedding the keyword default: (shape will be (None, 1, 50))
    keyword_embedding = Embedding(name = 'keyword_embedding',
                               input_dim = len(keyword_index),
                               output_dim = embedding_size)(keyword)
    
    # Merge the layers with a dot product along the second axis 
    # (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True,
                 axes = 2)([user_embedding, keyword_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # Squash outputs for classification
    out = Dense(1, activation = 'sigmoid')(merged)
    model = Model(inputs = [user, keyword], outputs = out)
    
    # Compile using specified optimizer and loss 
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    #print(model.summary())
    return model

model = build_embedding_model(embedding_size = 128, classification = False)


def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0):
    """Generate batches of samples for training. 
       Random select positive samples
       from pairs and randomly select negatives."""
    
    # Create empty array to hold batch
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        for idx, (user_id, keyword_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (user_id, keyword_id, 1)
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # Random selection
            random_user = random.randrange(len(user_index))
            random_keyword = random.randrange(len(keyword_index))
            #print(random_user, random_keyword)
            
            # Check to make sure this is not a positive example
            if (random_user, random_keyword) not in pairs:
                
                # Add to batch and increment index
                batch[idx, :] = (random_user, random_keyword, 0)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'user': batch[:, 0], 'keyword': batch[:, 1]}, batch[:, 2]
        
        
n_positive = len(pairs)
gen = generate_batch(pairs, n_positive, negative_ratio = 1)
# Train
h = model.fit_generator(gen, epochs = 50, steps_per_epoch = len(pairs) // n_positive)


# Extract embeddings
user_layer = model.get_layer('user_embedding')
user_weights = user_layer.get_weights()[0]


keyword_layer = model.get_layer('keyword_embedding')
keyword_weights = keyword_layer.get_weights()[0]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#PCA可视化

pca = PCA(n_components=2)
pca_result = pca.fit_transform(user_weights)
x=pca_result[:,0]
y=pca_result[:,1]
plt.scatter(x, y, color='r', marker='+')
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
index =[]
search = input("请输入搜索关键词：")
for j in range(len(x)):
    plt.annotate(txt1[j],xy=(x[j],y[j]),xytext=(x[j]+0.00005,y[j]+0.00005))
plt.show()
for i in range(len(k)):
    if (search in k[i]):
        index.append(i)
if len(index) > 1:
    for i in range(len(index)):
        print(k[index[i]])
if len(index) == 1:

    key = index[0]
    print("所搜索到的表格是",k[key])

#calculate cosine similarity 
    from sklearn.metrics.pairwise import cosine_similarity
    cos = np.array(cosine_similarity(user_weights)[key])
    recommendations = cos.argsort()[::-1][0:10]
    print(recommendations)
#print(user_weights[0].reshape(-1,1))
#print(cosine_similarity(user_weights[0].reshape(-1,1),user_weights[1].reshape(-1,1)))
    d={}
    num = 10
    for i in range(num):
    #d[i] = np.sum(np.abs(user_weights[key-1]-user_weights[recommendations[i]]))
        print(cos[recommendations[i]])
    for i in range(num):
    #d[i] = np.sum(np.abs(user_weights[key-1]-user_weights[recommendations[i]]))
        print(k[recommendations[i]])
if len(index) == 0:
    print("无法利用该关键词匹配到关联数据表，请尝试输入关联部门进行下一步检索")
    fun()



'''
#d1= np.sum(np.abs(user_weights[0]-user_weights[3]))
#d2= np.sum(np.abs(user_weights[0]-user_weights[2]))
#d3= np.sum(np.abs(user_weights[0]-user_weights[1]))
#d1 = np.dot(user_weights[0],user_weights[1])/(np.linalg.norm(user_weights[0])*(np.linalg.norm(user_weights[1])))
#print(d)
#print(d1)
#print(d2)
#print(d3)
#print(user_weights)
'''