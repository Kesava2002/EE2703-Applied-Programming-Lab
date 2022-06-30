from inspect import unwrap
import pylab as pl
import matplotlib.pyplot as pt
import numpy as np

# Initial Examples

def fft_modified(function,sampling_rate,x_limiter,tolerance,title,phasepoints=1):
    
    N = len(function)
    T = N/(sampling_rate)

    w_n = np. linspace(-sampling_rate*np.pi, sampling_rate*np.pi, N+1)
    w_n = w_n[:-1]

    f_n = pl.fftshift(pl.fft(function))/(len(function))
    
    # Plotting Magnitude and Phase Plots of the DFT

    pt.plot(w_n,np.abs(f_n),'r')
    pt.xlim(-x_limiter,x_limiter)
    pt.xlabel("Frequency")
    pt.ylabel("Magnitude of DFT")
    pt.title(f"DFT for {title}")
    pt.show()
    pt.plot(w_n,np.angle(f_n),'wo')

    # Finding the phase points at relevant points
    ii = np.where(abs(f_n)>tolerance)
    pt.plot(w_n[ii],np.angle(f_n[ii]),'go')
    if phasepoints == 1:
        for i, j in zip(w_n[ii], np.angle(f_n[ii])):
            if abs(j)< 1e-6:
                j=0
            pt.text(i, j+0.25, '({}, {})'.format(i, j))
    pt.xlabel("Frequency")
    pt.ylabel("Phase of DFT")
    pt.title(f"DFT for {title}")
    pt.xlim(-x_limiter,x_limiter)
    pt.show()
    if phasepoints == 2:
        print(f"Relevant Points in the Phase Plot for {title}")
        for i in range(len(ii)):
            print(w_n[ii[i]],np.angle(f_n[ii[i]]))   

# Sampling rate
sr = 128/(2*np.pi)             

# Spectrum of sin(5t)

t_1 = np.linspace(0,2*np.pi,129)
t_1 = t_1[:-1]

f_1 = np.sin(5*t_1)

fft_modified(f_1,sr,10, 1e-3,"sin(5t)")

# Spectrum of (1+0.1cost)cos10t

t_2 = np.linspace(-4*np.pi,4*np.pi,513)
t_2 = t_2[:-1]

f_2 = (1+0.1*np.cos(t_2))*np.cos(10*t_2)

fft_modified(f_2,sr,15,1e-3,"(1+0.1cost)cos(10t)",2)

# Question 2

# Spectrum of sin^3t

t_3 = np.linspace(-8*np.pi,8*np.pi,1025)
t_3 = t_3[:-1]
f_3 = np.sin(t_3)**3

pt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
fft_modified(f_3,sr,5,1e-3,"sin^3x")

# Spectrum of cos^3t

t_4 = np.linspace(-8*np.pi,8*np.pi,1025)
t_4 = t_4[:-1]
f_4 = np.cos(t_4)**3

pt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
fft_modified(f_4,sr,5,1e-3,"cos^3x")



# Question 3

# Spectrum of cos(20t+cos(5t))

t_5 = np.linspace(-8*np.pi,8*np.pi,1025)
t_5 = t_5[:-1]
f_5 = np.cos(20*t_5+5*np.cos(t_5))

fft_modified(f_5,sr,30,1e-3,f"cos(20t+cos(5t))",2)

# Question 4

def gauss(t):
    return np.exp(-0.5*(t**2))

# Continuous time Fourier transfrom of exp(-t^2/2) is 1/(2pi)^0.5 * exp(-w^2/2)

def actual_ft(t):
    return np.exp(-(t**2)/2)/np.sqrt(2*np.pi)

def estimated_dft(tolerance, samples, x_limit):
    T = 2*np.pi
    N = samples
    error = 1

    while error > tolerance:
        #print(error)
        tn = np.linspace( -T/2, T/2, N+1)
        tn = tn[:-1]

        wn = np. linspace(-N*np.pi/T, N*np.pi/T, N+1)
        wn = wn[:-1]

        fn = pl.fftshift(pl.fft(gauss(tn)))/(N*2*np.pi/T)

        error = np.sum(np.abs(np.abs(fn)-np.abs(actual_ft(wn))))

        T = T*2
        N = N*2
    print(f"True error : {np.sum(np.abs(np.abs(fn)-np.abs(actual_ft(wn))))}")  
    print(f"Samples = {N} and Time Period = {T}")  
    pt.plot(wn,abs(fn),'k')
    pt.xlim(-x_limit,x_limit)
    pt.xlabel("Frequency")
    pt.ylabel("Magnitude of DFT")
    pt.title(f"DFT for Gaussian function ")
    pt.show()

    pt.plot(wn,abs(actual_ft(wn)),'g')
    pt.title("Plot of Actual CTFT of the Gaussian Function")
    pt.xlabel("Frequency")
    pt.ylabel("CTFT Magnitude") 
    pt.xlim(-x_limit,x_limit)
    pt.show()
    
    # ii = np.where(abs(fn)<tolerance)
    # pt.plot(wn[ii],np.angle(fn[ii]),'go')
    # pt.xlabel("Frequency")
    # pt.ylabel("Phase of DFT")
    # pt.title(f"DFT for Gaussian function ")
    # pt.xlim(-x_limit,x_limit)
    # pt.show()  


estimated_dft(1e-6,128,10)

