########################################################
########################################################
########################################################
######### Work of Kesava Aruna Prakash R L #############
#################### EE20B061 ##########################
########################################################
########################################################
########################################################

#******************************************************#

import numpy as np
import math
import matplotlib.pyplot as pt
import scipy.integrate as sc

# f1(x)= e^x
# f2(x)= cos(cos(x))

# Defining functions using np module so that function returns an array of output if input is an array

def f1(input):
    return np.exp(input)
def f1p(input):
    return np.exp(input%(2*math.pi))    
def f2(input):
    return np.cos(np.cos(input))

plotx=np.linspace(-2*math.pi,4*math.pi,1000)

#******************************************************#

# Question 1

# Plots of actual functions 

pt.title("Plot of f1: Semilog")
pt.plot(plotx,f1(plotx),"-")
pt.plot(plotx,f1p(plotx),"-")
g=["e^x","e^(x%2pi)"] 
pt.grid()
pt.semilogy()
pt.xlabel("x")
pt.ylabel("f1(x)")
pt.legend(g)
pt.show()

pt.plot(plotx,f2(plotx),"-")
pt.title("Plot of f2")
pt.grid()
pt.xlabel("x")
pt.ylabel("f2(x)")
pt.show()

#******************************************************#

# Question 2

# Defining functions u and v, and also creating functions to return a and b coefficients

def u(x,k,func):
    if func=="f1":
        return f1(x)*np.cos(k*x)
    elif func=="f2":
        return f2(x)*np.cos(k*x)  

def v(x,k,func):
    if func=="f1":
        return f1(x)*np.sin(k*x)
    elif func=="f2":
        return f2(x)*np.sin(k*x) 


def an(func,n):
    a_coeff=np.empty(n)
    if func=="f1": 
        a_coeff[0]=sc.quad(u,0,2*math.pi,args=(0,'f1'))[0]/(2*math.pi)     
        for i in range(1,n):
            a_coeff[i]=sc.quad(u,0,2*math.pi,args=(i,'f1'))[0]/(math.pi)
    elif func=="f2":
        a_coeff[0]=sc.quad(u,0,2*math.pi,args=(0,'f2'))[0]/(2*math.pi)
        for i in range(1,n):
            a_coeff[i]=sc.quad(u,0,2*math.pi,args=(i,'f2'))[0]/(math.pi) 
    return a_coeff        

def bn(func,n):
    b_coeff=np.empty(n)
    if func=="f1":      
        for i in range(1,n):
            b_coeff[i]=sc.quad(v,0,2*math.pi,args=(i,'f1'))[0]/(math.pi)
    elif func=="f2":
        for i in range(1,n):
            b_coeff[i]=sc.quad(v,0,2*math.pi,args=(i,'f2'))[0]/(math.pi)
    return b_coeff        

# Getting the a and b coefficients

a_f1=an('f1',26)
b_f1=bn('f1',26)
a_f2=an('f2',26)
b_f2=bn('f2',26)

# Forming the Fourier coefficients column array 

temp= [a_f1[0]]
for i in range(len(b_f1)-1):
    temp.append(a_f1[i+1])
    temp.append(b_f1[i+1])        
coeff_f1=np.array(temp)        

temp1= [a_f2[0]]
for i in range(len(b_f2)-1):
    temp1.append(a_f2[i+1])
    temp1.append(b_f2[i+1])        
coeff_f2=np.array(temp1)        


x_axis=np.linspace(0,51,51)

# Plotting Fourier series coefficients

pt.plot(x_axis,np.abs(coeff_f1),'ro')
pt.title("SEMILOG PLOT OF FOURIER COEFFICIENTS OF f1")
pt.semilogy()
pt.show()

pt.plot(x_axis,np.abs(coeff_f1),'bo')
pt.title("LOGLOG PLOT OF FOURIER COEFFICIENTS OF f1")
pt.loglog()
pt.show()

pt.plot(x_axis,np.abs(coeff_f2),'ro')
pt.title("SEMILOG PLOT OF FOURIER COEFFICIENTS OF f2")
pt.semilogy()
pt.show()

pt.plot(x_axis,np.abs(coeff_f2),'bo')
pt.title("LOGLOG PLOT OF FOURIER COEFFICIENTS OF f2")
pt.loglog()
pt.show()

#******************************************************#

# Question 4

# Forming the M matrix

x=np.linspace(0,2*math.pi,400)

f1_values=f1(x)
f2_values=f2(x)

col1=np.ones(400).T
M=np.array(col1)
for i in range(1,26):
    col2=np.cos(i*x).T
    col3=np.sin(i*x).T
    M=np.c_[M,col2,col3]

#******************************************************#

# Question 5

# Finding the best fit Fourier coefficients

fit_f1=np.linalg.lstsq(M,f1_values,rcond=None)[0]  
fit_f2=np.linalg.lstsq(M,f2_values,rcond=None)[0]  

#******************************************************#

# Question 6

# Comparing the best fit Fourier coefficients with the ones calculated using integration

pt.title("Best fit coefficients and calculated coefficients for f1")
pt.plot(x_axis,np.abs(fit_f1),'ko')
pt.plot(x_axis,np.abs(coeff_f1),'yo')
r=["Obtained value","Calculated Value"]
pt.semilogy()
pt.legend(r)
pt.show()

# print(abs(fit_f1-coeff_f1))
# print(abs(fit_f2-coeff_f2))

pt.title("Best fit coefficients and calculated coefficients for f2")
pt.plot(x_axis,np.abs(fit_f2),'ko')
pt.plot(x_axis,np.abs(coeff_f2),'yo')
pt.semilogy()
pt.legend(r)
pt.show()

print("Maximum deviation for f1 :",max(np.abs(fit_f1-coeff_f1)))
print("Maximum deviation for f2 :",max(np.abs(fit_f2-coeff_f2)))
pt.title("Error in calculating coefficients")
pt.plot(x_axis,np.abs(fit_f1-coeff_f1),'go')
pt.plot(x_axis,np.abs(fit_f2-coeff_f2),'ko')
pt.semilogy()
pt.legend(["f1","f2"])
pt.show()



#******************************************************#

# Question 7

# Comparing function values obtained from best fit to actual function values

K_f1=np.matmul(M,fit_f1.T)
K_f2=np.matmul(M,fit_f2.T)

k=["Obtained value","Actual Value"]

pt.title("f1 values calculated from best fit compared with the actual f1 values")
pt.plot(x,np.abs(K_f1),'go')
pt.plot(x,np.abs(f1_values),'k-')
pt.semilogy()
pt.legend(k)
pt.show()

pt.title("f2 values calculated from best fit compared with the actual f2 values")
pt.plot(x,K_f2,'go')
pt.plot(x,f2_values,'k-')
pt.legend(k)
pt.show()

#******************************************************#



