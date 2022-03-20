import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# グラフの更新を行う関数updateを定義する
def update(fig, x, data, t, it, omega_list):
    fig.clear()
    # 上から順に3列グラフを作る
    ax0 = fig.add_axes([0.075, 0.66, 0.875, 0.26])
    ax1 = fig.add_axes([0.075, 0.36, 0.875, 0.26])
    ax2 = fig.add_axes([0.075, 0.06, 0.875, 0.26])
    
    ax2.set_xlabel("x")
    ax1.set_ylabel("u(x,t)")

    ax0.set_ylim(-0.5, 7.5)
    ax0.set_xlim(-15, 15)
    ax0.text(0.02, 0.85, 'ω=%.5g ' % omega_list[0], transform=ax0.transAxes, size=12)
    ax0.plot(x[0][it], data[0][it], color="blue", marker=None)
    ax1.set_ylim(-0.5, 7.5)
    ax1.set_xlim(-15, 15)
    ax1.text(0.02, 0.85, 'ω=%.5g ' % omega_list[1], transform=ax1.transAxes, size=12)
    ax1.plot(x[1][it], data[1][it], color="blue", marker=None)
    ax2.set_ylim(-0.5, 7.5)
    ax2.set_xlim(-15, 15)
    ax2.text(0.02, 0.85, 'ω=%.5g ' % omega_list[2], transform=ax2.transAxes, size=12)
    ax2.plot(x[2][it], data[2][it], color="blue", marker=None)

    time = str(math.floor(10*t[it])/10)
    ax0.set_title("t="+time)


# 初期値や計算範囲を入力する
N_t = 100
t_min = -3
t_max = 3
t_list = np.linspace(t_min, t_max, N_t)
delta_t = t_list[1]-t_list[0]
N_y = 750 # yはパラメータを表す
y_min = -30
y_max = 30
y = np.linspace( y_min, y_max, N_y)
omega_list = [1, 0.3, 0.05]
R_1 = 1
R_2 = 1
lambda_1 = 0.6
lambda_2 = 0.3
x_list_list = []
u_list_list = []


# 計算を行う
for omega in omega_list:
    kappa_1 = np.sqrt(1-omega*lambda_1)/2
    kappa_2 = np.sqrt(1-omega*lambda_2)/2
    x_list = []
    u_list = []
    for t in t_list:

        x0_1 = (1/(2*kappa_1))*np.log(R_1/(2*kappa_1))
        xi_1 = (2*kappa_1)*(-(y/np.sqrt(omega))+(2*t/lambda_1)+x0_1)
        x0_2 = (1/(2*kappa_2))*np.log(R_2/(2*kappa_2))
        xi_2 = (2*kappa_2)*(-(y/np.sqrt(omega))+(2*t/lambda_2)+x0_2)
        phi_1 = np.log((1-2*kappa_1)/(1+2*kappa_1))
        phi_2 = np.log((1-2*kappa_2)/(1+2*kappa_2))
        gamma_12 = np.log(((kappa_1-kappa_2)/(kappa_1+kappa_2))**2)
        f_p = 1+np.exp(xi_1-phi_1)+np.exp(xi_2-phi_2)+np.exp((xi_1-phi_1)+(xi_2-phi_2)+gamma_12)
        f_n = 1+np.exp(xi_1+phi_1)+np.exp(xi_2+phi_2)+np.exp((xi_1+phi_1)+(xi_2+phi_2)+gamma_12)
        f_pt = 4*(kappa_1/lambda_1)*(1+np.exp((xi_2-phi_2)+gamma_12))*np.exp(xi_1-phi_1) + 4*(kappa_2/lambda_2)*(1+np.exp((xi_1-phi_1)+gamma_12))*np.exp(xi_2-phi_2)
        f_nt = 4*(kappa_1/lambda_1)*(1+np.exp((xi_2+phi_2)+gamma_12))*np.exp(xi_1+phi_1) + 4*(kappa_2/lambda_2)*(1+np.exp((xi_1+phi_1)+gamma_12))*np.exp(xi_2+phi_2)

        u = ((f_pt*f_n)-(f_p*f_nt))/(f_n*f_p)
        x = y/(np.sqrt(omega))+np.log(f_p/f_n)

        x_list.append(x)
        u_list.append(u)
    x_list_list.append(x_list)
    u_list_list.append(u_list)


# グラフを描画し保存する
fig = plt.figure(figsize=(8, 7.5))
anim = FuncAnimation(fig, lambda it: update(fig, x_list_list, u_list_list, t_list, it, omega_list), len(t_list), interval=delta_t*1000, blit=False) # アニメを作る関数
anim.save('CH_soliton.gif', writer='pillow')