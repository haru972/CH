import numpy as np
import math
import scipy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# リストvecで表されるm*n Hankel行列の行列式を返す関数を定義する
def hankel_determinant(m, n, vec):
    if n == 0:
        return 1
    else:
        matrix = LA.hankel(vec[m:n+m], vec[n+m-1:2*n+m-1])
        return(LA.det(matrix))


# グラフの更新を行う関数updateを定義する
def update(fig, x, data, data_2, t, it):
    fig.clear()
    # 左に実際の波形を表すグラフax0、右にピークの位置を表すグラフax1を作る
    ax0 = fig.add_axes([0.075, 0.15, 0.55, 0.7])
    ax1 = fig.add_axes([0.7, 0.15, 0.28, 0.7])
    
    ax0.set_ylim(-11, 11)
    ax0.set_xlim(-20, 20)
    ax0.set_xlabel("x")
    ax0.set_ylabel("u(x,t)")
    ax0.plot(x, data[it], color="black", marker=None)

    for i in range(len(data_2)):
        ax1.plot(data_2[i], t, marker=None)
    ax1.plot([-20, 20], [t[it], t[it]], "-", color="black")
    ax1.set_ylim(-4, 4)
    ax1.set_xlim(-20, 20)
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")

    time = str(math.floor(10*t[it])/10)
    ax0.set_title("t="+time)


# 計算範囲を入力する
N_t = 200
t_min = -4 # tの絶対値が大きいとき挙動が不安定になるので注意する。
t_max = 4
t_list = np.linspace(t_min, t_max, N_t)
delta_t = t_list[1]-t_list[0]
N_x = 750
xmin = -20
xmax = 20
x = np.linspace(xmin, xmax, N_x)


# 初期値を入力する
N = 5 # ピーコンの個数
lambda_vec = [1.5, 1.2, 0.9, 0.5, -0.5]
R_0_vec = [1.0, 1.0, 1.0, 1.0, 1.0]
u_list = []
q_list = []
for i in range(N):
    q_list.append([]) # 先に空のリストをN個用意し、後で値を入れる


# p,q,u(x,t)を計算する
for t in t_list:
    # 行列の成分を表す数列を定義する
    hankel_vec = []
    for i in range(2*N+1):
        sum = 0
        for j in range(N):
            sum = sum+((lambda_vec[j])**i) *R_0_vec[j]*np.exp((2/lambda_vec[j])*t)
        hankel_vec.append(sum)


    p_vec = []
    q_vec = []
    for j in range(N):
        i = j+1 # 1からNまで動く変数

        # 先に分母を計算し、0となるときはエラーを出力する
        p_mother = (hankel_determinant(1, N-i+1, hankel_vec)*(hankel_determinant(1, N-i, hankel_vec)))
        q_mother = hankel_determinant(2, N-i, hankel_vec)

        if p_mother == 0:
            print("error_p", t, j)
            p = 0
        else:
            p = 4*hankel_determinant(0, N-i+1, hankel_vec)*hankel_determinant(2, N-i, hankel_vec)/(hankel_determinant(1, N-i+1, hankel_vec)*(hankel_determinant(1, N-i, hankel_vec)))

        if q_mother == 0:
            print("error_q", t, j)
            q = 0
            p = 0
        else:
            q = np.log(2*hankel_determinant(0, N-i+1, hankel_vec)/hankel_determinant(2, N-i, hankel_vec))
        
        p_vec.append(p)
        q_vec.append(q)
    

    # uにp*exp(|x-q|)を重ねる
    u = 0
    for i in range(N):
        u = u+p_vec[i]*np.exp(-np.abs(x-q_vec[i]))
    u_list.append(u)


    # qもグラフに必要なため保存する
    for i in range(N):
        q_list[i].append(q_vec[i])


# グラフを描画し保存する
fig = plt.figure(figsize=(8, 3))
anim = FuncAnimation(fig, lambda it: update(fig, x, u_list,q_list, t_list, it), len(t_list), interval=delta_t*1000, blit=False)#アニメを作る関数
anim.save('CH_peakon.gif', writer='pillow')