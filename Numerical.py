import numpy as np
from scipy.integrate import quad, dblquad

class Numerical:

    def __init__(self, f, df):
        '''Activation function and its derivative that are been used in the neural network'''
        self.f = f
        self.df = df

    def gauss_func(self, x, mu, sigma):
        '''Defining a gaussian density function'''
        return np.exp(-(x-mu)**2/(2*sigma**2)) / (sigma*(np.sqrt(2*np.pi))) 

    '''1) Show the simplest case of a single input, as it propergates through the network and
     how its *length* *ql* changes in the downstram layers'''
    def q_map_density(self, z, q0):
        return self.gauss_func(z, 0, 1) * (self.f(np.sqrt(q0)*z)**2)    

    def q_map(self, sw, sb, q0):
        return sw * quad(self.q_map_density, -np.inf, np.inf, args=(q0))[0] + sb

    def q_star(self, sw, sb , q0, maxL=10, tol=None):
        '''sw = variance of gaussian initialized weights
        sb = variance of gaussian initialized biases
        q0 = initial variance/lenth of the input signal
        maxL = number of layer to compute q_star
        tol = tolorence btw iteration 
        '''
        qs = np.array([q0])
        for l in range(maxL-1):
            qs1 = self.q_map(sw, sb, qs[-1])
            qs = np.append(qs,qs1)
            if tol and np.abs(qs[-1] - qs[-2]) < tol:
                    break
        return qs[-1], qs

    '''2) The case with two input, as it propergates through the network and
     how its correlation coefficient *c_ab* changes in the downstram layers'''
    def qab_map_density(self, z2, z1, qaa, qbb, c0):
        u1 = np.sqrt(qaa)*z1
        u2 = np.sqrt(qbb)*((c0)*(z1) + np.sqrt(1-(c0)**2)*(z2))
        return self.gauss_func(z1, 0, 1) * self.gauss_func(z2, 0, 1) * self.f(u1) * self.f(u2)

    def qab_map(self, sw, sb, qaa, qbb, c0):
        return sw * dblquad(self.qab_map_density, -np.inf, np.inf,
                        lambda x : -np.inf,
                        lambda x : np.inf,
                        args = (qaa, qbb, c0))[0] + sb
        
    def c_map(self, sw, sb, qaa, qbb, c0):
        qaa_nxt = self.q_map(sw, sb, qaa)
        qbb_nxt = self.q_map(sw, sb, qbb)
        qab_nxt = self.qab_map(sw, sb, qaa, qbb, c0)
        return qab_nxt / np.sqrt(qaa_nxt * qbb_nxt), qaa_nxt, qbb_nxt

    def c_star(self, sw, sb, qaa0, qbb0, c0, maxL=60, tol=None):
        '''
        sw = variance of gaussian initialized weights
        sb = variance of gaussian initialized biases
        qaa0 = initial variance/lenth of the first input signal
        qbb0 = initial variance/lenth of the second input signal
        c0 = correlation coefficient btw the two inputs
        maxL = number of layer to compute q_star
        tol = tolorence btw iteration
        '''
        cs = np.array([c0])
        qa = qaa0
        qb = qbb0
        for l in range(maxL-1):
            c, qa, qb = self.c_map(sw, sb, qa, qb, cs[-1])
            cs = np.append(cs, c)
            if tol and np.abs(cs[-1] - cs[-2]) < tol:
                    break
        return cs[-1], cs

    '''3) The stability of this fixed point depends on the slope of the map chi_1'''
    def chi1_density(self, z, q):
        return self.gauss_func(z, 0, 1) * (self.df(np.sqrt(q) * z))**2

    def sw_sb(self, q, chi1):
        sw = chi1 / quad(self.chi1_density, -np.inf, np.inf, args=(q))[0]
        sb = q - sw * quad(self.q_map_density, -np.inf, np.inf, args=(q))[0]
        return sw, sb

    def sw_sb_combo(self):    
        qrange = np.linspace(0, 3, 50)
        sw = np.array([])
        sb = np.array([])
        for q in qrange:
            w, b = self.sw_sb(q, 1)
            sw = np.append(sw, w)
            sb = np.append(sb, b)        
        return sw, sb

    '''4) Depth scale for correlations between inputs: xi_c'''
    def ch1_c_density(self, z1, z2, qaa, qbb, c0):
        u1 = np.sqrt(qaa)*z1
        u2 = np.sqrt(qbb)*((c0)*(z1) + np.sqrt(1-(c0)**2)*(z2))
        return self.gauss_func(z1, 0, 1) * self.gauss_func(z2, 0, 1) * self.df(u1) * self.df(u2)

    def xi_c(self, sw, sb, qaa0, qbb0, c0, maxL=60):
        '''
        sw = variance of gaussian initialized weights
        sb = variance of gaussian initialized biases
        qaa0 = initial variance/lenth of the first input signal
        qbb0 = initial variance/lenth of the second input signal
        c0 = correlation coefficient btw the two inputs
        maxL = number of layer to compute q_star
        '''
        qa_star = self.q_star(sw, sb , qaa0)[0]
        qb_star = self.q_star(sw, sb , qaa0)[0]
        cstar = self.c_star(sw, sb, qaa0, qbb0, c0)[0]
        chi_c = sw * dblquad(self.ch1_c_density, -np.inf, np.inf,
                            lambda x : -np.inf,
                            lambda x : np.inf,
                            args=(qa_star, qb_star, cstar))[0]
        return 1 / -np.log(chi_c)