# Adapted by Hendy: (in classroom's GDrive)
# Originally:
# http://bit.ly/sa2-python
# Long URL:
# http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

# coding: utf-8

# Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
# In[1]:
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# In[2]:
# define objective function


def f(x):
    x1 = x[0]
    x2 = x[1]
    obj = 0.2 + x1**2 + x2**2 - 0.1 * \
        math.cos(6.0*3.1415*x1) - 0.1*math.cos(6.0*3.1415*x2)
    return obj


# In[2]:
# Start location
x_start = [0.8, -0.5]
print("x_start=", x_start)

# In[3]:
# NOTE: THIS IS ONLY FOR DRAWING COUNTOUR PLOT VISUALIZATION
# TO HELP LEARNING, NOT PART OF SIMULATED ANNEALING ALGORITHM
# Design variables at mesh points
i1 = np.arange(-1.0, 1.0, 0.01)
i2 = np.arange(-1.0, 1.0, 0.01)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] = 0.2 + x1m[i][j]**2 + x2m[i][j]**2 \
            - 0.1*math.cos(6.0*3.1415*x1m[i][j]) \
            - 0.1*math.cos(6.0*3.1415*x2m[i][j])

# Create a contour plot
plt.figure()

# Specify contour lines
#lines = range(2,52,2)
# Plot contours
CS = plt.contour(x1m, x2m, fm)  # ,lines)
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
# Add some text to the plot
plt.title('Non-Convex Function')
plt.xlabel('x1')
plt.ylabel('x2')
# Start point is shown by ">" (play) marker
plt.plot(x_start[0], x_start[1], marker='>', markersize=20)

# In[6]:
##################################################
# Simulated Annealing
##################################################
# HYPERPARAMETERS
# Number of cycles
n: int = 50
# Number of trials per cycle
# m: int = 50
# Number of accepted solutions
n_accepted: int = 0
# Probability of accepting worse solution at the start
p1: float = 0.7
# Probability of accepting worse solution at the end
p50: float = 0.001
# Initial temperature
t1: float = -1.0/math.log(p1)
# Final temperature
t50: float = -1.0/math.log(p50)
# Fractional reduction of temperature every cycle
frac: float = (t50/t1)**(1.0/(n-1.0))
print('n=', n)
# print('m=', m)
print('na=', n_accepted)
print('p1=', p1)
print('p50=', p50)
print('t1=', t1)
print('t50=', t50)
print('frac=', frac)

# In[7]:
# Initialize x
x = np.zeros((n+1, 2))
x[0] = np.copy(x_start)
# xi = np.zeros(2)
xi = np.copy(x_start)
n_accepted = n_accepted + 1
# Current accepted results so far
xc = np.zeros(2)
xc = np.copy(x[0])
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = fc
# Best results
best_x = np.copy(xc)
best_f = fc
# Current temperature
t = t1
# DeltaE Average
# DeltaE_avg = 0.0

# print('x=', x)
print('xi=', xi)
print('na=', n_accepted)
print('xc=', xc)
print('fc=', fc)
print('fs=', fs)
print('t=', t)
# print('DeltaE_avg=', DeltaE_avg)

# In[8]: MAIN SIMULATED ANNEALING ALGORITHM
for i in range(n):
    print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))

    # Generate new trial points
    xi[0] = xc[0] + random.random() - 0.5
    xi[1] = xc[1] + random.random() - 0.5
    # Clip to upper and lower bounds
    xi[0] = max(min(xi[0], 1.0), -1.0)
    xi[1] = max(min(xi[1], 1.0), -1.0)
    DeltaE = f(xi) - fc
    print("  New state (%f, %f), f(xi)=%f. Current state f(c)=%f. ΔE=%s" %
        (xi[0], xi[1], f(xi), fc, DeltaE))
    if DeltaE >= 0:
        # Initialize DeltaE_avg if a worse solution was found
        #   on the first iteration
        # if (i==0 and j==0): DeltaE_avg = DeltaE
        # objective function is worse
        # generate probability of acceptance
        p = math.exp(-DeltaE / t)
        # determine whether to accept worse point
        therandom = random.random()
        if (therandom < p):
            # accept the worse solution
            print("    With p=%f, random is %f, ACCEPT worse solution" % (p, therandom))
            accept = True
        else:
            # don't accept the worse solution
            print("    With p=%f, random is %f, REJECT worse solution" % (p, therandom))
            accept = False
    else:
        # objective function is lower, automatically accept
        # THEN SET AS BEST
        accept = True
    if accept == True:
        # update currently accepted solution
        xc[0] = xi[0]
        xc[1] = xi[1]
        fc = f(xc)
        # increment number of accepted solutions
        n_accepted = n_accepted + 1
        # update DeltaE_avg
        # DeltaE_avg = (DeltaE_avg * (na-1.0) + DeltaE) / na

    # record current state
    x[i+1][0] = xc[0]
    x[i+1][1] = xc[1]
    fs[i+1] = fc

    # Record the best x values at the end of every cycle
    if fc < best_f:
        print('  NEW Best-So-Far (%f, %f). f(x)=%f' % (xc[0], xc[1], fc))
        best_x = np.copy(xc)
        best_f = fc
    else:
        print('  UNCHANGED Best-So-Far (%f, %f). f(x)=%f' % (x[i][0], x[i][1], fs[i]))

    # Lower the temperature for next cycle
    t = frac * t

# print solution
print('Best solution: ', best_x)
print('Best objective: ', best_f)

# In[9]: Plot all cycles
plt.plot(x[:, 0], x[:, 1], 'y-o')
# Start point is shown by ">" (play) marker
print("Start:", x_start)
plt.plot(x_start[0], x_start[1], marker='>', markersize=10)
# Stop point is shown by "s" (stop) marker
print("Stop:", xi)
plt.plot(xi[0], xi[1], marker='s', markersize=10)
# Best point is shown by "*" (star) marker
print("Best:", best_x)
plt.plot(best_x[0], best_x[1], marker='*', markersize=20)
plt.savefig('contour.png')

# In[9]: Plot iteration objectives and trials
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(fs, 'r.-')
ax1.legend(['Objective'])

ax2 = fig.add_subplot(212)
ax2.plot(x[:, 0], 'b.-')
ax2.plot(x[:, 1], 'g--')
ax2.legend(['x1', 'x2'])

# Save the figure as a PNG
plt.savefig('iterations.png')

plt.show()


#%%
