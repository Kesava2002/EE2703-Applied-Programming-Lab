import numpy as np
import sympy as spy
import scipy.signal as sg
import matplotlib.pyplot as pt

# Question 1

s = spy.symbols('s')

# Function to get output of low-pass filter in s-domain


def lowpass(R1, R2, C1, C2, G, Vi):
    A = spy.Matrix([[0, 0, 1, -1/G], [-1/(1+s*R2*C2), 1, 0, 0],
                   [0, -G, G, 1], [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = spy.Matrix([0, 0, 0, -Vi/R1])
    V = A.inv()*b
    return A,b,V

A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]

# Converting the transfer function to be used in signals toolbox as a LTI system 

def ltisys(v):
    num,denom = spy.fraction(v)
    num_coeff = spy.poly(num).all_coeffs()
    denom_coeff = spy.poly(denom).all_coeffs()
    num_coeff = [float(x) for x in num_coeff]
    denom_coeff = [float(x) for x in denom_coeff]

    return sg.lti(num_coeff,denom_coeff)
    
H_LP = ltisys(Vo)

# Bode plots for the Transfer function of the circuit

w,H_mag,H_phase = H_LP.bode()

pt.plot(w,H_mag,'k')
pt.title("Bode Magnitude Plot of the Low Pass Filter")
pt.xlabel("Frequency")
pt.semilogx()
pt.ylabel("Magnitude")
pt.show()

pt.plot(w,H_phase,'k')
pt.title("Bode Phase Plot of the Low Pass Filter")
pt.xlabel("Frequency")
pt.semilogx()
pt.ylabel("Phase")
pt.show()



# Step response for the circuit

t = np.linspace(0,0.005,10000)

u_t = np.ones([10000])

t,y_t,svec = sg.lsim(H_LP,u_t,t)

pt.title("Step Response of the Low Pass filter")
pt.ylabel("Output Voltage Vo")
pt.xlabel("Time")
pt.plot(t,y_t,'r')
pt.show()

# Question 2

x_t = np.sin(2000*np.pi*t)+np.cos(2*10**6*np.pi*t) 

t,Vo_LP,svec = sg.lsim(H_LP,x_t,t)

pt.title("Output of the Low Pass filter for input Vi(t)")
pt.ylabel("Output Voltage Vo")
pt.xlabel("Time")
pt.plot(t,Vo_LP,'v')
pt.show()

# Question 3

# High Pass Filter

def highpass(R1, R3, C1, C2, G, Vi):
    A = spy.Matrix([[0, 0, 1, -1/G], [-s*C2*R3/(1+s*R3*C2), 1, 0, 0],
                   [0, -G, G, 1], [1/R1+s*C1+s*C2, -s*C2, 0, -1/R1]])
    b = spy.Matrix([0, 0, 0, s*C1*Vi])
    V = A.inv()*b
    return A,b,V

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo_HP = V[3]

H_HP = ltisys(Vo_HP)

# Bode plots for the Transfer function of the High Pass Filter

w,H_mag,H_phase = H_HP.bode()

pt.plot(w,H_mag,'k')
pt.title("Bode Magnitude Plot of the High Pass Filter")
pt.xlabel("Frequency")
pt.semilogx()
pt.ylabel("Magnitude")
pt.show()

pt.plot(w,H_phase,'k')
pt.title("Bode Phase Plot of the High Pass Filter")
pt.xlabel("Frequency")
pt.semilogx()
pt.ylabel("Phase")
pt.show()

# Question 4

# For a decay of 100 second^-1 and frequency of 1e6 * 2pi(High) 

t1 = np.linspace(0,2e-5,10000)
vi_t_1 = 1*np.exp(-100*t1)*np.sin(2e6*np.pi*t1)

t1,Vo_HP_1,svec  = sg.lsim(H_HP,vi_t_1,t1)

pt.title("Output of the High Pass Filter for input(Damped Sinusoid)-High Frequency")
pt.ylabel("Output Voltage Vo")
pt.xlabel("Time")
pt.plot(t1,Vo_HP_1,'r')
#pt.show()

# For a decay of 100 second^-1 and frequency of 1e3 * 2pi(Low) 

vi_t1_2 = 1*np.exp(-100*t1)*np.sin(2e3*np.pi*t1)

t1,Vo_HP_2,svec  = sg.lsim(H_HP,vi_t1_2,t1)

pt.title("Output of the High Pass Filter for input(Damped Sinusoid)-Low Frequency")
pt.ylabel("Output Voltage Vo")
pt.xlabel("Time")
pt.plot(t1,Vo_HP_2,'k')
leg = ["1MHz","1KHz"]
pt.legend(leg)
pt.show()

# Question 5

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo_HP = V[3]
H_HP = ltisys(Vo_HP)
t,Vo_HP_Step = sg.impulse(H_HP,None,t)

pt.title("Step Response of the High Pass filter")
pt.ylabel("Output Voltage Vo")
pt.xlabel("Time")
pt.plot(t,Vo_HP_Step,'g')
pt.show()


