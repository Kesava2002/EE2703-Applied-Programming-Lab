from inspect import unwrap
import pylab as pl
import matplotlib.pyplot as pt
import numpy as np

# Initial Examples

t_1 = np.linspace(0,2*np.pi,129)
t_1 = t_1[:-1]

f_1 = np.sin(5*t_1)
w_1 = np.linspace(-64,63,128)

def fft_modified(function,w,x,y,title,phase):
    f = pl.fftshift(pl.fft(function))/len(function)  
    pt.plot(w,abs(f),'r')
    pt.xlim(-x,x)
    pt.xlabel("Frequency")
    pt.ylabel("Magnitude of DFT")
    pt.title(f"DFT for {title}")
    pt.show()
    pt.plot(w,np.angle(f),'wo')
    ii = np.where(abs(f)>y)
    pt.plot(w[ii],np.angle(f[ii]),'go')
    if phase == 1:
        for i, j in zip(w[ii], np.angle(f[ii])):
            if abs(j)< 1e-6:
                j=0
            pt.text(i, j+0.25, '({}, {})'.format(i, j))
    pt.xlabel("Frequency")
    pt.ylabel("Phase of DFT")
    pt.title(f"DFT for {title}")
    pt.xlim(-x,x)
    pt.show()
    if phase == 2:
        print(f"Relevant Points in the Phase Plot for {title}")
        for i in range(len(ii)):
            print(f"({w[ii[i]]},{np.angle(f[ii[i]])}")    

fft_modified(f_1,w_1,10,1e-3,"Sine wave of Angular frequency = 5 rad/s",1)

t_2 = np.linspace(-4*np.pi,4*np.pi,513)
t_2 = t_2[:-1]

f_2 = (1+0.1*np.cos(t_2))*np.cos(10*t_2)
w_2 = np.linspace(-64,64,513)
w_2 = w_2[:-1]

fft_modified(f_2,w_2,15,1e-3,"Cosine Waves with multiple angular frequencies",2)

# Question 2

# Spectrum of sin^3t

t_3 = np.linspace(-8*np.pi,8*np.pi,1025)
t_3 = t_3[:-1]
f_3 = np.sin(t_3)**3
w_3 = np.linspace(-64,64,1025)
w_3 = w_3[:-1]

pt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
fft_modified(f_3,w_3,5,1e-3,f"$sin^3x$",1)

# Spectrum of cos^3t

t_4 = np.linspace(-8*np.pi,8*np.pi,1025)
t_4 = t_4[:-1]
f_4 = np.cos(t_4)**3
w_4 = np.linspace(-64,64,1025)
w_4 = w_4[:-1]

pt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
fft_modified(f_4,w_4,5,1e-3,f"$cos^3x$",1)

# Question 3

t_5 = np.linspace(-8*np.pi,8*np.pi,1025)
t_5 = t_5[:-1]
f_5 = np.cos(20*t_5+5*np.cos(t_5))
w_5 = np.linspace(-64,64,1025)
w_5 = w_5[:-1]

fft_modified(f_5,w_5,30,1e-3,f"cos(20t+cos5t)",2)

# Question 4

def gauss(t):
    return np.exp(-0.5*(t**2))

# Continuous time Fourier transfrom of exp(-t^2/2) is 1/(2pi)^0.5 * exp(-w^2/2)

def actual_ft(t):
    return np.exp(-(t**2)/2)/np.sqrt(2*np.pi)

def estimated_dft(tolerance, samples, x):
    T = 2*np.pi
    N = samples
    error = 1

    while error > tolerance:
        #print(error)
        t_n = np.linspace( -T/2, T/2, N+1)
        t_n = t_n[:-1]

        w_n = np. linspace(-N*np.pi/T, N*np.pi/T, N+1)
        w_n = w_n[:-1]

        f_n = pl.fftshift(pl.fft(gauss(t_n)))/(N*2*np.pi/T)

        error = np.sum(np.abs(np.abs(f_n)-np.abs(actual_ft(w_n))))

        T = T*2
        N = N*2
    print(f"True error : {np.sum(np.abs(np.abs(f_n)-np.abs(actual_ft(w_n))))}")  
    print(f"Samples = {N} and Time Period = {T/np.pi}")  
    pt.plot(w_n,abs(f_n),'r')
    pt.xlim(-x,x)
    pt.xlabel("Frequency")
    pt.ylabel("Magnitude of DFT")
    pt.title(f"DFT for Gaussian function ")
    pt.show()
    ii = np.where(abs(f_n)<tolerance)
    pt.plot(w_n[ii],np.angle(f_n[ii]),'go')
    pt.xlabel("Frequency")
    pt.ylabel("Phase of DFT")
    pt.title(f"DFT for Gaussian function ")
    pt.xlim(-x,x)
    pt.show()  


estimated_dft(1e-6,128,10)

