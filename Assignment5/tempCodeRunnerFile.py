# Finding best fit for all values (fit1)

y_fit = np.log(errors)
x_fit = np.c_[n.T,n1.T]
B,logA = np.linalg.lstsq(x_fit,y_fit,rcond=None)[0]
A=np.exp(logA)

# Finding best fit for N= 500 to N_iterations (fit2)

y_fit1 = np.log(errors[500:])
x_fit1 = np.c_[n[500:].T,n1[500:].T]
#print(x_fit1)
B1,logA1 = np.linalg.lstsq(x_fit1,y_fit1,rcond=None)[0]
A1=np.exp(logA1)