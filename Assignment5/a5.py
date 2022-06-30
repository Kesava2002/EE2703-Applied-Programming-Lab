########################################################
########################################################
########################################################
######### Work of Kesava Aruna Prakash R L #############
#################### EE20B061 ##########################
########################################################
########################################################
########################################################

import sys
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as pt

# Default variables
Nx = 25 
Ny = 25
radius = 8
N_iteration = 1500

if len(sys.argv)==5:
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = int(sys.argv[3])
    N_iteration=int(sys.argv[4])
elif len(sys.argv)==1:
    print("Default arguments taken")
else:
    print("Expected arguments: Nx, Ny, Radius and number of iterations")
    exit(0)        


# Creating matrix for phi
phi = np.zeros([Nx,Ny])

# Since we know the plate is 1cm x 1cm, we can split the sides into Nx and Ny points each such that the middle point is origin
x = np.linspace(-0.5,0.5,Nx)
y = np.linspace(-0.5,0.5,Ny)
#y = np.flip(y)

# Forming the meshgrid
Y,X = np.meshgrid(y,x)

# Conditions for checking
X1,Y1=np.where(Y*Y+X*X<=(radius/(Nx-1)+0.01)**2)
for i in range(len(X1)):
    phi[X1[i]][Y1[i]]=1

# Contour plot of phi after setting up 1V in the middle
pt.title("CONTOUR PLOT OF PHI")
cnt= pt.contour(x,y,phi)
norm= matplotlib.colors.Normalize(vmin=0, vmax=1)
sm = pt.cm.ScalarMappable(norm=norm, cmap = cnt.cmap)
sm.set_array([])
pt.colorbar(sm, ticks=cnt.levels)
pt.plot(x[X1],y[Y1],'rx')
pt.show()       

errors=np.empty(N_iteration)

# Iterating for N_iteration times and recalculating phi at each point in the iteration
for i in range(N_iteration):
    oldphi=phi.copy()
    phi[1:-1,1:-1]=0.25*(phi[1:-1,2:]+phi[1:-1,0:-2]+phi[0:-2,1:-1]+phi[2:,1:-1])

    # Boundary conditions
    phi[1:-1,0]=phi[1:-1,1]
    phi[1:-1,-1]=phi[1:-1,-2]
    phi[-1,:]=phi[-2,:]

    # Setting phi as 1V in the centre region
    for j in range(len(X1)):
        phi[X1[j]][Y1[j]]=1
    
    # Calculating errors
    errors[i]=(abs(phi-oldphi)).max()


n = np.linspace(1,N_iteration,N_iteration)
n1= np.ones(N_iteration)

pt.title("Plot of Error vs n (Every 50 points): LOGLOG plot")
pt.plot(n[::50],errors[::50],'o--')
pt.xlabel("N(every 50 points)")
pt.ylabel("Error")
pt.loglog()
pt.show()

pt.title("Plot of Error vs n (Every 50 points): SEMILOG plot")
pt.plot(n[::50],errors[::50],'ko--')
pt.xlabel("N(every 50 points)")
pt.ylabel("Error")
pt.semilogy()
pt.show()

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

pt.title("Plots of Fit1, Fit2 and actual Error: SEMILOG plot")
pt.plot(errors,'r')
pt.plot(A*np.exp(B*n),'g')
pt.plot(n[500:],A1*np.exp(B1*n[500:]),'k')
pt.semilogy()
pt.xlabel("N")
pt.ylabel("Error")
leg=["Error","Fit1","Fit2"]
pt.legend(leg)
pt.show()

def cumulative_error(x):
    return -A/B*np.exp(B*(x+0.5))

# Plot of cumulative error

pt.loglog(n[100::100],cumulative_error(n[100::100]),'ro')
pt.xlabel("No. of iterations")
pt.ylabel("Cumulative Error")
pt.title("Cumulative Error in log-log plot (every 100th points)")
pt.show()

pt.title(f"CONTOUR PLOT OF POTENTIAL : {N_iteration} iterations")
cntplot=pt.contour(x,y,phi)
pt.clabel(cntplot)
norm1= matplotlib.colors.Normalize(vmin=0, vmax=1)
sm1 = pt.cm.ScalarMappable(norm=norm1, cmap = cntplot.cmap)
sm1.set_array([])
pt.colorbar(sm1, ticks=cntplot.levels)
pt.plot(x[X1],y[Y1],'ro')
pt.show()


fig1=pt.figure()
ax=p3.Axes3D(fig1,auto_add_to_figure=False)
ax.set_title("The 3-D surface plot of the potential")
fig1.add_axes(ax)
#pt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=pt.cm.jet)
norm2= matplotlib.colors.Normalize(vmin=0, vmax=1)
sm2 = pt.cm.ScalarMappable(norm=norm2, cmap = surf.cmap)
sm2.set_array([])
pt.colorbar(sm2,shrink=0.9)
fig1.set_size_inches(6, 6)
pt.show()

# Calculating current density
Jx = 0.5*(phi[1:-1,0:-2]-phi[1:-1,2:])
Jy = 0.5*(phi[0:-2,1:-1]-phi[2:,1:-1])

pt.title("CURRENT DENSITY PLOT")
pt.quiver(x[1:-1],y[1:-1],Jx,Jy)
pt.plot(x[X1],y[Y1],'ro')
pt.show()