'''手敲HMM进行分词'''

import math

infinite = float(-2**31)

def log_normalize(a):#对a做正则化：先归一化(该数的值在s(sum)中所占比例),再取对数，最后就等于logx-logsum
    s = 0
    for x in a:
        s += x
    if s == 0:
        print("Error from log_normalize")
        return
    s = math.log(s)
    for i in range(len(a)):
        if a[i]==0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i])-s




def mle():  #0B/1M/2E/3S    最大似然估计法
    pi = [0] * 4    # npi[i]：i状态的个数
    a = [[0] * 4 for x in range(4)]     # na[i][j]：从i状态到j状态的转移个数
    b = [[0] * 65536 for x in range(4)] # nb[i][o]：从i状态到o字符的个数   shape为(4,65536)  unicode编码65536个
    with open("24.pku_training.utf8","r",encoding="utf-8") as f:
        data = f.read()[3:]
    tokens = data.split("  ")   #去掉文件头，留下真正的数据
    last_q = 2  #上一个词是end
    iii = 0
    old_progress = 0

    print("进度")
    for k,token in enumerate(tokens):      #enumerate(tokens)循环tokens并包含一个index list
        progress = float(k)/float(len(tokens))  #表示进度
        if progress>old_progress+0.1:           #表示进度
            print("%.3f"%progress)               #表示进度
            old_progress = progress             #表示进度

        token = token.strip()
        n = len(token)  #词中包含字的个数
        if n<=0:
            continue
        if n==1:        #n=1，说明是3S(single)类型
            pi[3] += 1
            a[last_q][3] +=1    # A中上一个词的结束(last_q)到当前状态(3S)多了一次
            b[3][ord(token[0])] +=1     #从状态3S到观测值token[0]多了一次   ord(token[0])将该单独的字转为unicode编码
            last_q = 3      ##上一个词就变成了3S
            continue
        '''n>=2的情况'''
        # 初始向量π		n!=1，则begin和end各加1
        pi[0] += 1      #begin多一次
        pi[2] += 1      #end多一次
        pi[1] += (n-2)     #n-2为middle词的数量，加上

        # 转移矩阵A
        a[last_q][0] += 1   #上一个状态到begin多了一个
        last_q = 2  #上一个词成了end
        if n==2:
            a[0][2] += 1    #begin到end多了一个
        else:   #n>2    则该词至少有三个字
            a[0][1] += 1    #begin到middle+1
            a[1][1] += (n-3)    ##middle到middle加的数为n-begin数-end数-一个middle
            a[1][2] += 1    #middle到end+1

        #发射矩阵B
        b[0][ord(token[0])] +=1     #从begin到第一个字+1
        b[2][ord(token[n-1])] +=1  #从end到最后一个字+1
        for i in range(1,n-1):  #左闭右开
            b[1][ord(token[i])] += 1    #从middle到中间的字+1

    #正则化
    log_normalize(pi)
    for i in range(4):
        log_normalize(a[i])
        log_normalize(b[i])

    return pi,a,b


def list_write(pi_file, v):
    with open(pi_file,"a") as f:
        for a in v:
            f.write(str(a))
            f.write(" ")
        f.write("\n")

def viterbi(pi,A,B,o):   #维特比算法那一页中只要是乘的，在这里都用加，因为π，A，B用log处理过了
    T = len(o)  #观测序列
    delta = [[0 for i in range(4)] for t in range(T)]   #len(o)行，4列
    pre = [[0 for i in range(4)] for t in range(T)] # 前一个状态   # pre[t][i]：t时刻的i状态，它的前一个状态是多少

    for i in range(4):
        delta[0][i] = pi[i] + B[i][ord(o[0])]   #初始化   第一个时刻(字)状态为i的概率分别为多少
    for t in range(1,T):    #递推，对每一个字（观测值）
        for i in range(4):   #对四种状态分别求
            delta[t][i] = delta[t-1][0] + A[0][i]   #t-1时刻状态begin到状态i
            for j in range(1,4):                    #t-1时刻状态middle，end，single分别到状态i，取一个最大的情况
                vj = delta[t-1][j] + A[j][i]    #对应公式中t-1时刻的delta j 乘以 状态j到状态i的转移概率
                if delta[t][i] <vj:
                    delta[t][i] = vj
                    pre[t][i] = j      #确定t-1时刻的状态为j
            delta[t][i] += B[i][ord(o[t])]  #找到该时刻该状态的最大值
    decode = [-1 for t in range(T)]     # 解码：回溯查找最大路径
    q = 0
    for i in range(1,4):    #找到最后时刻的隐状态
        if delta[T-1][i] > delta[T-1][q]:
            q = i
    decode[T-1] = q
    for t in range(T-2,-1,-1):  #从倒数第二个开始倒着往回直到第一个
        q = pre[t+1][q]     #利用后一时刻的隐状态推出前一时刻的隐状态
        decode[t] = q
    return decode   #返回最大路径


def segment(sentence, decode):
    N = len(sentence)
    i = 0
    while i<N:  #B/M/E/S
        if decode[i] == 0 or decode[i] == 1:       #Begin/middle/end
            j = i + 1
            while j < N:
                if decode[j] == 2 or decode[j] == 3:  #遇到了end/single
                    break
                j += 1
            print(sentence[i:j+1]+"|")
            i = j + 1
        elif decode[i]==3 or decode[i]==2:    #single
            print(sentence[i:i+1]+"|")
            i += 1
        else:
            print("Error",i,decode[i])
            i += 1


def save_parameter(pi,A,B):
    pi_file = "pi.txt"
    list_write(pi_file,pi)

    a_file = "A.txt"
    for a in A:
        list_write(a_file,a)

    b_file = "B.txt"
    for b in B:
        list_write(b_file,b)

def list_write(pi_file, v):
    with open(pi_file,"a") as f:
        for a in v:
            f.write(str(a))
            f.write(" ")
        f.write("\n")


if __name__=="__main__":
    pi,A,B = mle()  #得到pi,A,B
    print(pi)

    # save_parameter(pi,A,B)  #保存参数
    print("训练完成")
    with open("24.MyBook.txt","r",encoding="utf-8") as f:
        data = f.read().strip()
    decode = viterbi(pi,A,B,data)   #维特比算法求出结果
    print(decode)
    decode[0] = 0
    segment(data,decode)    #根据最大路径切分
