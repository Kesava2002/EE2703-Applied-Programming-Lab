
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as pt

def f_Transfer(decay):
    num = np.poly1d([1,decay])
    denom = num**2+2.25
    return num, denom

def f_t(t,decay,freq):
    return np.cos(freq*t)*np.exp(-1*decay*t)

# Question 1

num1,denom1 = f_Transfer(0.05)

f = np.poly1d([1,0,2.25])
H1 = sp.lti(num1,f*denom1)

t = np.linspace(0,50,1001)

t1,h1 = sp.impulse(H1,None,t)
pt.title("Time Response of Spring for decay = 0.05")
pt.xlabel("t")
pt.ylabel("x")
pt.plot(t1,h1)
pt.show()

# Question 2

num2,denom2 = f_Transfer(0.5)
H2 = sp.lti(num2,f*denom2)

t2,h2 = sp.impulse(H2,None,t)
pt.title("Time Response of Spring for decay = 0.5")
pt.plot(t2,h2)
pt.xlabel("t")
pt.ylabel("x")
pt.show()

# Question 3

# Transfer function of X(s) can be derived from x''(t)+ 2.25x(t) = u(t) 

X = sp.lti([1],[1,0,2.25])

w = np.arange(1.4,1.6,0.05)

leg = []
for i in w:
    ft = f_t(t,0.05,i)
    t,y,svec = sp.lsim(X,ft,t)
    pt.plot(t,y)
    leg.append(f"w= {i}")
pt.title("Time Response of Spring for different freqencies")
pt.legend(leg)
pt.xlabel("t")
pt.ylabel("x")
pt.show()   

# Question 4

# x'' + x = y
# Xs^2 - x(0)s - x'(0) + X = Y

# y'' +2y = 2x
# Ys^2 - y(0)s - y'(0) + 2Y = 2X

X_s = sp.lti([1,0,2],[1,0,3,0])
Y_s = sp.lti([2],[1,0,3,0])

tn = np.linspace(0,20,201)
tx,x = sp.impulse(X_s,None,tn)
ty,y = sp.impulse(Y_s,None,tn)

pt.plot(tx,x)
pt.plot(ty,y)
pt.xlabel("t")
pt.ylabel("Position")
pt.title("Time Response for Coupled Strings")
leg1 = ["x","y"]
pt.legend(leg1)
pt.show()

# Question 5

R = 100 
L = 1e-6
C = 1e-6
H_RLC = sp.lti(1,[L*C,R*C,1])  # Transfer function of system
w,mag_RLC,phi_RLC = H_RLC.bode()

pt.title("Bode Plot of Transfer function of RLC circuit: Magnitude ")
pt.xlabel("w")
pt.plot(w,mag_RLC,'r')
pt.semilogx()
pt.ylabel("Magnitude")
pt.show()

pt.title("Bode Plot of Transfer function of RLC circuit: Phase ")
pt.xlabel("w")
pt.plot(w,phi_RLC,'r')
pt.semilogx()
pt.ylabel("Phase")
pt.show()

# Question 6

t_large = np.arange(0,1e-2,1e-7)
t_small = np.arange(0,30e-6,1e-7)
vi = lambda t: np.cos((10**3)*t)-np.cos((10**6)*t) 

tl,vo_large,svec = sp.lsim(H_RLC,vi(t_large),t_large)
pt.title("Plot of Output Voltage : 0 to 10ms")
pt.xlabel("Time")
pt.ylabel("Voltage")
pt.plot(tl,vo_large)
pt.show()

ts,vo_small,svec = sp.lsim(H_RLC,vi(t_small),t_small)
pt.title("Plot of Output Voltage : 0 to 30us")
pt.xlabel("Time")
pt.ylabel("Voltage")
pt.plot(ts,vo_small)
pt.show()

