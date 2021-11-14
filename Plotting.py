import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Experiment import Experment
from Numerical import Numerical

def d_tanh(x):
    return 1. / np.cosh(x)**2

def exp_c_star(sigma_w, sigma_b, x1, x2):
    Exp = Experment(sigma_w, sigma_b)
    cab = np.absolute(Exp.cl(1, x1, x2, 1000).numpy())
    for i in range(3,118,2):
        cab = np.append(cab, np.absolute(Exp.cl(i, x1, x2, 1000)))
    return cab

def exp_c_plot(x1, x2):
    plt.plot(range(59), exp_c_star(6.25,0.09,x1,x2), color='green', label='$\sigma_w = 2.5$')
    plt.plot(range(59), exp_c_star(16,0.09,x1,x2), color='blue', label='$\sigma_w = 4$')
    plt.plot(range(59), exp_c_star(3,0.09,x1,x2), color='red', label='$\sigma_w = 1.7$')
    plt.plot(range(59), exp_c_star(3,0.09,x1,x2), color='red')
    plt.plot(range(59), exp_c_star(6.25,0.09,x1,x2), color='green')
    plt.plot(range(59), exp_c_star(16,0.09,x1,x2), color='blue')
    plt.plot(range(59), exp_c_star(16,0.09,x1,x2), color='blue')
    plt.plot(range(59), exp_c_star(3,0.09,x1,x2), color='red')
    plt.plot(range(59), exp_c_star(6.25,0.09,x1,x2), color='green')
    plt.legend()
    plt.savefig('/images/exp_c_plot.png')


def num_c_plot():
    num = Numerical(np.tanh, d_tanh)
    plt.plot(np.linspace(0,60,60),num.c_star(6.25,0.09,11,2,0.2)[1],color = 'green', label = '$\sigma_w$ = 2.5')
    plt.plot(np.linspace(0,60,60),num.c_star(6.25,0.09,11,2,0.5)[1],color = 'green')
    plt.plot(np.linspace(0,60,60),num.c_star(6.25,0.09,11,2,0.8)[1],color = 'green')
    
    plt.plot(np.linspace(0,60,60),num.c_star(16,0.09,11,2,0.2)[1],color = 'blue', label = '$\sigma_w$ = 4')
    plt.plot(np.linspace(0,60,60),num.c_star(16,0.09,11,2,0.5)[1],color = 'blue')
    plt.plot(np.linspace(0,60,60),num.c_star(16,0.09,11,2,0.8)[1],color = 'blue')
    
    plt.plot(np.linspace(0,60,60),num.c_star(3,0.09,11,2,0.2)[1],color = 'red', label = '$\sigma_w$ = 1.7')
    plt.plot(np.linspace(0,60,60),num.c_star(3,0.09,11,2,0.5)[1],color = 'red')
    plt.plot(np.linspace(0,60,60),num.c_star(3,0.09,11,2,0.8)[1],color = 'red')
    
    plt.xlabel('$L$', fontsize=16)
    plt.ylabel('$c$', fontsize=16)
    plt.legend(loc='best')
    plt.savefig('/images/num_c_plot.png')


def exp_q_star(sigma_w, sigma_b, x1, x2, x3):
    Exp = Experment(sigma_w, sigma_b)
    q1 = (tf.tensordot(x1 ,x1, [[1],[1]])/ 784).eval(session=tf.compat.v1.Session())
    for i in range(1, 58, 2):
        q1 = np.append(q1, Exp.ql(i, x1, 1000)[0])

    q2 = (tf.tensordot(x2 ,x2, [[1],[1]])/784).eval(session=tf.compat.v1.Session())
    for i in range(1, 58, 2):
        q2 = np.append(q2, Exp.ql(i, x2, 1000)[0]) 

    q3 = (tf.tensordot(x3 ,x3, [[1],[1]])/784).eval(session=tf.compat.v1.Session())
    for i in range(1, 58, 2):
        q3 = np.append(q3, Exp.ql(i, x2, 1000)[0])    

    return q1, q2, q3

def exp_q_plot(x1, x2, x3):
    plt.plot(range(30), exp_q_star(3, 0.3, x1, x2, x3)[0], color='green', label='$\sigma_w = 3$')
    plt.plot(range(30), exp_q_star(3, 0.3, x1, x2, x3)[1], color='green')
    plt.plot(range(30), exp_q_star(3, 0.3, x1, x2, x3)[2], color='green')
    plt.plot(range(30), exp_q_star(4, 0.3, x1, x2, x3)[0], color='red', label='$\sigma_w = 4$')
    plt.plot(range(30), exp_q_star(4, 0.3, x1, x2, x3)[1], color='red')
    plt.plot(range(30), exp_q_star(4, 0.3, x1, x2, x3)[2], color='red')
    plt.plot(range(30), exp_q_star(2, 0.3, x1, x2, x3)[0], color='blue', label='$\sigma_w = 2$')
    plt.plot(range(30), exp_q_star(2, 0.3, x1, x2, x3)[1], color='blue')
    plt.plot(range(30), exp_q_star(2, 0.3, x1, x2, x3)[2], color='blue')
    plt.xlabel('$L$')
    plt.ylabel('$q$')
    plt.legend(loc = 1)
    plt.savefig('/images/exp_q_plot.png')


def num_q_plot():
    num = Numerical(np.tanh, d_tanh)
    plt.plot(np.linspace(0,10,10),num.q_star(6.25,0.09,4)[1],color = 'green', label = '$\sigma_w$ = 2.5')
    plt.plot(np.linspace(0,10,10),num.q_star(6.25,0.09,6)[1],color = 'green')
    plt.plot(np.linspace(0,10,10),num.q_star(6.25,0.09,10)[1],color = 'green')

    plt.plot(np.linspace(0,10,10),num.q_star(16,0.09,4)[1],color = 'blue', label = '$\sigma_w$ = 4')
    plt.plot(np.linspace(0,10,10),num.q_star(16,0.09,6)[1],color = 'blue')
    plt.plot(np.linspace(0,10,10),num.q_star(16,0.09,10)[1],color = 'blue')

    plt.plot(np.linspace(0,10,10),num.q_star(3,0.09,4)[1],color = 'red', label = '$\sigma_w$ = 1.7')
    plt.plot(np.linspace(0,10,10),num.q_star(3,0.09,6)[1],color = 'red')
    plt.plot(np.linspace(0,10,10),num.q_star(3,0.09,10)[1],color = 'red')
    
    plt.xlabel('$L$', fontsize=16)
    plt.ylabel('$q$', fontsize=16)
    plt.legend(loc=1)
    plt.savefig('images/num_q_plot.png')


def phase_plot():
    num = Numerical(np.tanh, d_tanh)
    areaLabels=['Ordered','Chaotic']
    fig, ax = plt.subplots()
    ax.stackplot(num.sw_sb_combo()[0], num.sw_sb_combo()[1])

    ax.text(1.5, 0.25, areaLabels[0])


    ax.text(2.75,0.15, areaLabels[1])

    plt.xlim(1, 3.4)
    plt.ylim(0, 0.4)
    plt.xlabel('$\sigma_w^2$', fontsize=16)
    plt.ylabel('$\sigma_b^2$', fontsize=16)
    plt.legend()
    plt.savefig('/images/phase_plot.png')

if __name__ == '__main__':
    #phase_plot()
    #num_q_plot()
    #num_c_plot()
    #exp_q_plot()
    #exp_c_plot()
    x1 = tf.random.normal([1,784], mean=0.0, stddev=3.162)
    x2 = tf.random.normal([1,784], mean=0.0, stddev=2.449)
    x3 = tf.random.normal([1,784], mean=0.0, stddev=2)
    exp_q_plot(x1, x2, x3)


