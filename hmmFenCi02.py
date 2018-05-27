


def load_train():
    pi = [0] * 4

    with open("pi.txt","r") as f:
        contents = f.read().strip().split(" ")
        for i,content in enumerate(contents):
            pi[i] = content


    A = [[0] * 4  for x in range(4)]
    with open("A.txt","r") as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            numbers = line.strip().split(" ")
            for k,number in enumerate(numbers):
                A[i][k] = number
            i += 1

    B = [[0] * 65536 for x in range(4)]
    with open("B.txt","r") as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            numbers = line.strip().split(" ")
            for k,number in enumerate(numbers):
                B[i][k] = number
            i += 1

    return pi,A,B


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
            print(sentence[i:j+1],"|")
            i = j + 1
        elif decode[i]==3 or decode[i]==2:    #single
            print(sentence[i:i+1],"|")
            i += 1
        else:
            print("Error",i,decode[i])
            i += 1





if __name__=="__main__":
    pi,A,B = load_train()   #读出π，A，B

    # print(A)
    # print("\n")
    # print(B)
    with open("24.novel.txt","r",encoding="utf-8") as f:
        data = f.read().strip()
    decode = viterbi(pi,A,B,data)   #维特比算法求出结果
    print(decode)
    decode[0] = 0
    segment(data,decode)    #根据最大路径切分

