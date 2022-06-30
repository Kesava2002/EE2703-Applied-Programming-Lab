import pylab as pl
import numpy as np
import matplotlib.pyplot as pt
from scipy import randn

# Spectrum for sin(2^0.5t)

t_eg1 = np.linspace(-np.pi,np.pi,65)
t_eg1 = t_eg1[:-1]

dt_eg1 = t_eg1[1]-t_eg1[0]
fmax_eg1 = 1/dt_eg1

y_eg1 = np.sin(np.sqrt(2)*t_eg1)
#print(y_eg1)
y_eg1 = pl.fftshift(y_eg1)
#print(y_eg1)
y_eg1[32] = 0

y_eg1 = pl.fftshift(pl.fft(y_eg1))/64.0
w_eg1 = np.linspace(-np.pi*fmax_eg1,np.pi*fmax_eg1,65)
w_eg1 = w_eg1[:-1]

pt.plot(w_eg1,abs(y_eg1),'r')
pt.title("Spectrum of sin((2^0.5)t)")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

pt.plot(w_eg1,np.angle(y_eg1),'ro')
pt.title("Spectrum of sin((2^0.5)t)")
pt.xlabel("Phase")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

# Windowing Function

def windowfn(N):
    n = np.arange(N)
    wnd = pl.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
    return wnd

# Using windowing function on sin((2^0.5)t)    


t_eg2 = t_eg1
dt_eg2 = t_eg2[1]-t_eg2[0]
fmax_eg2 = 1/dt_eg2
w_eg2 = np.linspace(-np.pi*fmax_eg2,np.pi*fmax_eg2,65)
w_eg2 = w_eg2[:-1]

y_eg2 = np.sin(np.sqrt(2)*t_eg2)*windowfn(64)
y_eg2 = pl.fftshift(y_eg2)
y_eg2[32] = 0
Y_eg2 = pl.fftshift(pl.fft(y_eg2))/64.0

pt.plot(t_eg2,np.sin(np.sqrt(2)*t_eg2)*windowfn(64),'ro')
pt.title("Plot of sin((2^0.5)t) after Windowing")
pt.plot(t_eg2,np.sin(np.sqrt(2)*t_eg2),'go')
pt.legend(["With Windowing","Without Windowing"])
pt.show()

pt.plot(w_eg2,abs(Y_eg2),'r')
pt.title("Spectrum of sin((2^0.5)t): After Windowing")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

pt.plot(w_eg2,np.angle(Y_eg2),'ro')
pt.title("Spectrum of sin((2^0.5)t): After Windowing")
pt.xlabel("Phase")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

# Using a different time window

t_eg3 = np.linspace(-4*np.pi,4*np.pi,257)
t_eg3 = t_eg3[:-1]

dt_eg3 = t_eg3[1]-t_eg3[0]
fmax_eg3 = 1/dt_eg3
w_eg3 = np.linspace(-np.pi*fmax_eg3,np.pi*fmax_eg3,257)
w_eg3 = w_eg3[:-1]

y_eg3 = np.sin(np.sqrt(2)*t_eg3)*windowfn(256)
y_eg3 = pl.fftshift(y_eg3)
y_eg3[128] = 0
Y_eg3 = pl.fftshift(pl.fft(y_eg3))/256.0

pt.plot(w_eg3,abs(Y_eg3),'r')
pt.title("Spectrum of sin((2^0.5)t): Larger Timeframe")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

pt.plot(w_eg3,np.angle(Y_eg3),'ro')
pt.title("Spectrum of sin((2^0.5)t): Larger Timeframe")
pt.xlabel("Phase")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()


# Question 2

# Spectrum of cos^3(w_ot) with w_o = 0.86

t_q2 = np.linspace(-8*np.pi,8*np.pi,513)
t_q2 = t_q2[:-1]

dt_q2 = t_q2[1]-t_q2[0]
fmax_q2 = 1/dt_q2
w_q2 = np.linspace(-np.pi*fmax_q2,np.pi*fmax_q2,513)
w_q2 = w_q2[:-1]

w_o = 0.86
y_q2 = np.cos(w_o*t_q2)**3
y_q2 = pl.fftshift(y_q2)
Y_q2 = pl.fftshift(pl.fft(y_q2))/512.0

pt.plot(w_q2,abs(Y_q2),'r')
pt.title("Spectrum of cos^3(w_ot): Without Windowing")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

pt.plot(w_q2,np.angle(Y_q2),'ro')
pt.xlim(-10,10)
pt.show()

w_o = 0.86
y_q2 = (np.cos(w_o*t_q2)**3)*windowfn(512)
y_q2 = pl.fftshift(y_q2)
Y_q2 = pl.fftshift(pl.fft(y_q2))/512.0

pt.plot(w_q2,abs(Y_q2),'r')
pt.title("Spectrum of cos^3(w_ot): After Windowing")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-10,10)
pt.show()

pt.plot(w_q2,np.angle(Y_q2),'ro')
pt.xlim(-10,10)
pt.show()

# Question 3

# Function to estimate frequency and phase of the given input data

def estim_w_del(y):
    Y_new = pl.fftshift(pl.fft(y))/128.0
    w = np.linspace(-64,64,129)
    w = w[:-1]

    ii = np.where(w>0)

    omega = sum(abs(Y_new[ii])**2*w[ii])/sum(abs(Y_new[ii])**2)
    
    iii = abs(w-omega).argmin()
    phase = np.angle(Y_new[iii])
    if phase < 0:
        phase = phase+np.pi
             

    return omega,phase

t = np.linspace(-np.pi,np.pi,129)
t = t[:-1]
sample = np.cos(1.5*t+0.4)*windowfn(128)

print(estim_w_del(sample))

# Question 4

# Function to estimate frequency and phase of the given input noisy data

sample_noisy = np.cos(1.4*t+0.7)*windowfn(128)+0.1*np.random.randn(len(t))*windowfn(128)

print(estim_w_del(sample_noisy))

# Question 5

t_q4 = np.linspace(-np.pi,np.pi,1025)
t_q4 = t_q4[:-1]

y_q4 = np.cos(16*(1.5*t_q4)+8*t_q4**2/np.pi)*windowfn(1024)

y_q4 = pl.fftshift(y_q4)
Y_q4 = pl.fftshift(pl.fft(y_q4))/1024.0

w_q4 = np.linspace(-512,512,1025)
w_q4 = w_q4[:-1]

pt.plot(w_q4,abs(Y_q4),'r')
pt.title("Spectrum of Chirped Signal: After Windowing")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-60,60)
pt.show()

pt.plot(w_q4,np.angle(Y_q4),'ro')
pt.xlim(-100,100)
pt.show()

t_q5 = np.linspace(-np.pi,np.pi,1025)
t_q5 = t_q5[:-1]

y_q5 = np.cos(16*(1.5*t_q4)+8*t_q4**2/np.pi)

y_q5 = pl.fftshift(y_q5)
Y_q5 = pl.fftshift(pl.fft(y_q5))/1024.0

w_q5 = np.linspace(-512,512,1025)
w_q5 = w_q5[:-1]

pt.plot(w_q5,abs(Y_q5),'r')
pt.title("Spectrum of Chirped Signal: Without Windowing")
pt.xlabel("Magnitude")
pt.ylabel("Angular Frequency")
pt.xlim(-60,60)
pt.show()

# Question 6

def splitdft(y):
    y = np.split(y,16)
    dft = []
    for x in y:
        x = pl.fftshift(x)
        X = pl.fftshift(pl.fft(x))/64
        dft.append(X)
    return dft    

y_q5 = np.cos(16*(1.5*t_q4)+8*t_q4**2/np.pi)*windowfn(1024)
values = splitdft(y_q5)
values = np.array(values)


fig = pt.figure()
ax = fig.add_subplot(111, projection='3d')

t = np.linspace(-np.pi,np.pi,1025);t=t[:-1]
fmax = 1/(t[1]-t[0])
t = t[::64]
w = np.linspace(-fmax*np.pi,fmax*np.pi,64+1);w=w[:-1]
t,w = np.meshgrid(t,w)
surf = ax.plot_surface(w,t,abs(values).T,cmap=pt.cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
pt.ylabel("Frequency")
pt.xlabel("Time")
pt.title("Frequency Time plot of the DFT")
pt.show()

fig = pt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(w,t,np.angle(values).T,cmap=pt.cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
pt.ylabel("Frequency")
pt.xlabel("time")

pt.show()











