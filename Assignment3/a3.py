########################################################
########################################################
########################################################
# Work of Kesava Aruna Prakash R L #####################
######### EE20B061 #####################################
########################################################
########################################################
########################################################

import numpy as np
import scipy.special as sc
import pylab
import matplotlib.pyplot as pt
import scipy as scp
from statistics import *

try :
    fobj=np.loadtxt("fitting.dat",dtype='float')
    # Loading the data to a numpy array using loadtext
except OSError:
    print("Couldn't open file/ or error reading the file")
    exit()   
#print(fobj)

# Function to generate function f(t)
def g(t, A, B):
    return A*sc.jn(2,t)+B*t


# Question 3

# Generating sigma values

sigma=np.logspace(-1,-3,9)

# Printing data from 'fitting.dat'

pt.plot(fobj[:,0],fobj[:,1:])
r = []
for i in range(len(sigma)):
    r.append(f"$\sigma${i+1} = {sigma[i]}")
   


# Question 4    

#Plot without noise

pt.plot(fobj[:,0],g(fobj[:,0],1.05,-0.105),color='k') 
r.append("TRUE VALUE") 
pt.legend(r)
pt.ylabel("g(t,1.05,-0.105)+Noise")
pt.xlabel("TIME")
pt.show()

#Question 5

# Printing the errorbars

h=[]
pt.plot(fobj[:,1])
h.append("SIGMA=0.1")
k=np.arange(0,101,5)
pt.plot(k,fobj[::5,1])
h.append("DOWNSAMPLED DATA")
pt.plot(g(fobj[:,0],1.05,-0.105),color='k')
h.append("TRUE VALUE")
pt.errorbar(k,fobj[::5,1],sigma[0],fmt='ro')
h.append("ERROR")
pt.legend(h)
pt.show()

# Question 6

# Creating the column arrays and forming M matrix

g_array=np.zeros([len(fobj[:,1]),2])

for i in range(len(fobj[:,1])):
    g_array[i][0]=sc.jn(2,fobj[i][0])
    g_array[i][1]=fobj[i][0]

AB_array=np.zeros([2,1])

AB_array[0][0]=1.05
AB_array[1][0]=-0.105

temp=np.matmul(g_array,AB_array)
tempsum=0
for i in range(0,100):
    tempsum+=temp[i]-g(fobj[i][0],1.05,-0.105)
    #print(g(fobj[i][0],1.05,-0.105))
#print(tempsum)    

# Question 7

# Error matrix

min=100000
i_ind=-1
j_ind=-1
e=np.zeros([21,21])
a_array=np.arange(0,2.1,0.1)
b_array=np.arange(-0.2,0.01,0.01)
for i in range(len(a_array)):
    for j in range(len(b_array)):
        for k in range(len(fobj[:,0])):
            e[i][j]+=((fobj[k][1]-g(fobj[k][0],a_array[i],b_array[j]))**2)/101
        if e[i][j]<min:
            min=e[i][j]
            i_ind=i
            j_ind=j

# Question 8

# Plotting the contour

cntr =pt.contour(a_array,b_array,e,levels=20)
pt.clabel(cntr)
pt.xlabel("A")
pt.ylabel("B")
pt.show()
#print(min)
#print(a_array[i_ind])
#print(b_array[j_ind])

# Question 9

# Linear fit

p,w,q,z=scp.linalg.lstsq(g_array,fobj[:,1])

# A and B for best fit
#print("A and B values for the best fit:",p[0],p[1])

# Residual Error of the best fit
#print("Residual error of the best fit :",w)

# Question 10

# Function to return the A and B for each fit

def lstsq(input):
    return [np.linalg.lstsq(g_array,input)[0][0], np.linalg.lstsq(g_array,input)[0][1]]

# Function to return the MSerror with respect to (1.05,-0.105) after creating a new data set    
def manydata():
    error_a =[]
    error_b =[]
    for s in sigma:  
        a_fit = []
        b_fit = []
        for i in range(1000): 
            yy = np.random.normal(scale=s, size = (101))
            obj = lstsq(yy+g(fobj[:,0],1.05,-0.105))
            a_fit.append(obj[0])
            b_fit.append(obj[1])
        error_a.append(np.square(np.subtract(1.05,a_fit)/1.05).mean())   
        error_b.append(np.square(np.subtract(-0.105,b_fit)/-0.105).mean())
 
    return [error_a, error_b]
mse = manydata()
pt.plot(sigma,mse[0],'o--')  
pt.plot(sigma,mse[1],'o--')
pt.ylabel("ERROR OF A AND B")
pt.xlabel("SIGMA")
pt.show()   
        
# Question 11

# Plotting the errorbars

pt.errorbar(sigma,mse[0],fmt='ro--')
pt.errorbar(sigma,mse[1],fmt='ko--')
leg=["A Error","B Error"]
pt.legend(leg)
pt.title(" LOG-LOG plot of Error in A and B vs Sigma")
pt.ylabel("ERROR IN A AND B")
pt.xlabel("SIGMA")
pt.loglog()
pt.show()
      


