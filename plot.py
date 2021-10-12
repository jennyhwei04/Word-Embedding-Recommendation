import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(figsize=(6, 6))
#ax.set(aspect=1.0/ax.get_data_ratio()*x)
#选择ax1
plt.subplot(221)
#折线图
#x = [0.2,0.4,0.6,0.8]#点的横坐标
#x = ['REST','LAPT','AUTO']
x = np.arange(4)
plt.xticks([0,1,2,3],['20%','40%','60%','80%'])
#my_y_ticks = np.arange(0, 1, 0.5)
#plt.yticks(my_y_ticks)
k1 = [0.92,0.7,0.54,0.53]#线1的纵坐标
k2 = [1.7,1.18,0.81,0.56]#线2的纵坐标
k3 = [0.789,0.656,0.562,0.50]
k4 = [0.89,0.789,0.702,0.614]
plt.plot(x,k1,'s-',color = 'r',label="ED-MF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MF")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="ED-UserCF")#s-:方形
plt.plot(x,k4,'.-',color = 'y',label="UserCF")#o-:圆形
plt.xlabel("Train Rate")#横坐标名字
plt.ylabel("GovernmentData-RMSE")#纵坐标名字
#plt.ylim((0,4))
plt.legend(loc=1)
plt.subplot(222)
x = np.arange(4)
plt.xticks([0,1,2,3],['20%','40%','60%','80%'])
#my_y_ticks = np.arange(0, 1, 0.5)
#plt.yticks(my_y_ticks)
k1 = [0.92,0.67,0.47,0.46]#线1的纵坐标
k2 = [1.6,1,0.66,0.50]#线2的纵坐标
k3 = [0.6,0.52,0.4,0.26]
k4 = [0.71,0.61,0.5,0.36]
plt.plot(x,k1,'s-',color = 'r',label="ED-MF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MF")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="ED-UserCF")#s-:方形
plt.plot(x,k4,'.-',color = 'y',label="UserCF")#o-:圆形
plt.xlabel("Train Rate")#横坐标名字
plt.ylabel("GovernmentData-MAE")#纵坐标名字
#plt.ylim((0,4))
plt.legend(loc=1)
plt.subplot(223)
#折线图
#x = [0.2,0.4,0.6,0.8]#点的横坐标
#x = ['REST','LAPT','AUTO']
x = np.arange(4)
plt.xticks([0,1,2,3],['20%','40%','60%','80%'])
my_y_ticks = np.arange(0, 4, 1)
plt.yticks(my_y_ticks)
k1 = [1.1,1.03,1.01,0.95]#线1的纵坐标
k2 = [3.2,1.95,1.28,1]#线2的纵坐标
k3 = [3.36,3.17,3.04,2.90]
k4 = [3.54,3.38,3.23,3.08]
plt.plot(x,k1,'s-',color = 'r',label="ED-MF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MF")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="ED-UserCF")#s-:方形
plt.plot(x,k4,'.-',color = 'y',label="UserCF")#o-:圆形
plt.xlabel("Train Rate")#横坐标名字
plt.ylabel("Movielens-RMSE")#纵坐标名字
#plt.ylim((0,4))
plt.legend(loc=1)
plt.subplot(224)
x = np.arange(4)
plt.xticks([0,1,2,3],['20%','40%','60%','80%'])
my_y_ticks = np.arange(0, 4, 1)
plt.yticks(my_y_ticks)
k1 = [0.87,0.81,0.8,0.76]#线1的纵坐标
k2 = [3.06,1.88,1.13,0.77]#线2的纵坐标
k3 = [3.21,3.05,2.8,2.6]
k4 = [3.36,3.20,3.06,2.87]
plt.plot(x,k1,'s-',color = 'r',label="ED-MF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MF")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="ED-UserCF")#s-:方形
plt.plot(x,k4,'.-',color = 'y',label="UserCF")#o-:圆形
plt.xlabel("Train Rate")#横坐标名字
plt.ylabel("Movielens-MAE")#纵坐标名字
#plt.ylim((0,4))
plt.legend(loc=1)
fig.tight_layout()
plt.show()

#柱状图
'''
import numpy as np
import matplotlib.pyplot as plt
ED_MF = [3.54,3.38,3.23,3.08]
MF = [3.54,3.38,3.23,3.08]
#x = ['REST','LAPT','AUTO']
x = np.arange(4) #总共有几组，就设置成几，我们这里有三组，所以设置为3
total_width, n = 0.8, 4    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x, User_CF, color = "r",width=width,label='User_CF')
plt.bar(x + width, Item_CF, color = "y",width=width,label='Item_CF')
plt.bar(x + 2 * width, UserED_CF , color = "c",width=width,label='UserED_CF')
plt.bar(x + 3 * width, ItemED_CF , color = "g",width=width,label='ItemED_CF')
plt.xlabel("Algorithms")
plt.ylabel("RMSE")
plt.legend(loc = "best")
plt.xticks([0,1,2,3],['20%','40%','60%','80%'])
my_y_ticks = np.arange(0.8, 0.95, 0.02)
plt.ylim((0.8, 0.95))
plt.yticks(my_y_ticks)
plt.show()
'''