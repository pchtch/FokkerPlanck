import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#============================================
# Partial differential equations:
# Diffusion problem.
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
#============================================
# Solver for a tridiagonal matrix.
# a,b,c are the lower, center, and upper diagonals,
# r is the RHS vector.
def tridiag(a,b,c,r):
    n    = b.size
    gam  = np.zeros(n)
    u    = np.zeros(n)
    bet  = b[0]
    u[0] = r[0]/bet
    for j in range(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        if (bet == 0.0):
            print('[tridiag]: matrix not invertible.')
            exit()
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    for j in range(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

#============================================
# Driver for the actual integrators. Sets the initial conditions
# and generates the support point arrays in space and time.
# input: J      : number of spatial support points
#        dt0    : timestep
#        minmaxx: 2-element array containing minimum and maximum of spatial domain
#        minmaxt: 2-element array, same for time domain
#        fINT   : integrator (one of ftcs, implicit, cranknicholson)
#        fBNC   : boundary condition function
#        fINC   : initial condition function
def diffusion_solve(J,minmaxx,dt0,minmaxt,fINT,fBNC,fINC,**kwargs):
    k = 1  # Boltzmann constant
    T = 1           # Temperature in Kelvin
    b = 1/(k * T)
   
    # time and space discretization
    N  = int((minmaxt[1]-minmaxt[0])/dt0)+1
    dt = (minmaxt[1]-minmaxt[0])/float(N-1) # recalculate, to make exact
    dx = (minmaxx[1]-minmaxx[0])/float(J)
    x  = minmaxx[0]+(np.arange(J)+0.5)*dx
    t  = minmaxt[0]+np.arange(N)*dt
    D = 1.0
   
    # alpha factor
    alpha    = D*dt/(dx**2)
   
   
    print('[diffusion_solve]: alpha = %13.5e' % (alpha))
    print('[diffusion_solve]: N     = %7i' % (N))
    y        = fINT(x,t,alpha,fBNC,fINC)
   
   
    return x,t,y

#--------------------------------------------
# Forward-time centered-space integrator.
# Returns the full solution array (including
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.

def ftcs(x, t, alpha, fBNC, fINC):
    k = 1  # Boltzmann constant
    T = 1  # Temperature in Kelvin
    b = 1 / (k * T)
   
    J = x.size
    N = t.size
    W = np.zeros((J+2, N))
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    W[1:(J + 1), 0] = fINC(x)  # Set initial condition
   
    def F(x):
        a0, a1, a2, a3, a4 = 300, -0.38, 1.37, -2, 1
        return a0*k*T*(4*a4*x**3 + 3*a3*x**2 + 2*a2*x + a1)

    def dFdx(x):
        a0, a1, a2, a3, a4 = 300, -0.38, 1.37, -2, 1
        return a0*k*T*(12*a4*x**2 + 6*a3*x + 2*a2)

    # First step parameters
    def beta(x):
        return float(-b*dt*F(x)/(2*dx) + alpha)

    def gamma(x):
        return float(b*dt*dFdx(x) - 2*alpha)

    def delta(x):
        return float(b*dt*F(x)/(2*dx) + alpha)
   
    # Second set of parameters after first step
    def epsilon(x):
        return float(2*beta(x))
   
    def zeta(x):
        return float(2*dt*b*dFdx(x) - 4*alpha)
   
    def eta(x):
        return float(2*delta(x))

    # print(f"Time 0: {W[:, 0]}")

    # First step
    while True:
        for j in range(1, J + 1):
            W[j, 1] =  beta(j * dx) * W[j-1, 0] + gamma(j * dx) * W[j, 0] + delta(j * dx) * W[j+1, 0] + W[j, 0]
        break

    # Next time steps
    for n in range(2, N):
        W[0, n] = fBNC(0, W[:, n-1])
        W[J + 1, n] = fBNC(J + 1, W[:, n-1])  
        for j in range(1, J + 1):  
            epsilon_j = epsilon(j*dx)
            zeta_j = zeta(j*dx)
            eta_j = eta(j*dx)
           
            W[j, n] = W[j, n-2] + epsilon_j * W[j-1, n-1] + zeta_j * W[j, n-1] + eta_j * W[j+1, n-1]
   

    return W[1:(J+1),:]

#--------------------------------------------
# Fully implicit integrator.
# Returns the full solution array (including
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
# Uses tridiag to solve the tridiagonal matrix.
def implicit(x, t, alpha, fBNC, fINC):
    k = 1  # Boltzmann constant
    T = 1  # Temperature in Kelvin
    b = 1 / (k * T)
   
    J = x.size
    N = t.size
    W = np.zeros((J+2, N))
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Initial condition
    W[1:(J + 1), 0] = fINC(x)

    # Potential function and its derivative
        
    def F(x):
        a0, a1, a2, a3, a4 = 300, -0.38, 1.37, -2, 1
        return a0 * k * T * (4 * a4 * x**3 + 3 * a3 * x**2 + 2 * a2 * x + a1)

    def dFdx(x):
        a0, a1, a2, a3, a4 = 300, -0.38, 1.37, -2, 1
        return a0 * k * T * (12 * a4 * x**2 + 6 * a3 * x + 2 * a2)

    # Step parameters
    def beta(x):
        return float(b*dt*F(x)/(2*dx) - alpha)

    def gamma(x):
        return float(-b*dt*dFdx(x) + 2*alpha + 1)

    def delta(x):
        return float(-b*dt*F(x)/(2*dx) - alpha)
   
    # Parameter arrays for implicit scheme
    beta_j = np.array([beta(j * dx) for j in range(1, J+1)])
    gamma_j = np.array([gamma(j * dx) for j in range(1, J+1)])
    delta_j = np.array([delta(j * dx) for j in range(1, J+1)])
   
    # Iterate over time steps
    for n in range(1, N):
        # Update boundary conditions at each time step
        W[0, n] = fBNC(0, W[:, n-1])
        W[J + 1, n] = fBNC(1, W[:, n-1])
       
        # Solve the tridiagonal system for this time step
        r = W[1:J + 1, n - 1]  # Initialize r with values from previous time step
        W[1:J + 1, n] = tridiag(beta_j, gamma_j, delta_j, r)
   
    return W[1:J + 1, :]
 
#-------------------------------------------
# Initialization and Boundary Condition Functions
def Tconst(x):
    return np.zeros(x.size) + 1.0

def Tnorm(x):
    sigma = 0.01
    mu = 0.5
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def Bdirichlet(iside, y):
    if iside == 0:
        return -y[1]
    else:
        return -y[y.size - 2]
   

def init(solver,problem):
    if (solver == 'ftcs'):
        fINT = ftcs
    elif (solver == 'implicit'):
        fINT = implicit
    else:
        print('[init]: invalid solver %s' % (solver))
       
    if (problem == 'normal'):
        fINC    = Tnorm
        fBNC    = Bdirichlet
        minmaxx = np.array([0,1.0])
        minmaxt = np.array([0.0,1.0])
       
    if (problem == 'const'):
        fINC    = Tconst
        fBNC    = Bdirichlet
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,20.0])
       
    return fINT,fBNC,fINC,minmaxx,minmaxt

#--------------------------------------------
# Main execution function
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J", type=int, help="number of spatial support points (including boundaries)")
    parser.add_argument("dt", type=float, help="timestep")
    parser.add_argument("solver", type=str, help="Fokker-Planck solver:\n"
                                                 "    ftcs    : forward-time centered-space\n"
                                                 "    implicit: fully implicit\n"
                                                 "    CN      : Crank-Nicholson")
    parser.add_argument("problem", type=str, help="initial conditions:\n"
                                                  "    const   : constant\n"
                                                  "    norm    : normal distribution\n")

    args = parser.parse_args()
    J = args.J
    dt = args.dt
    solver = args.solver
    problem = args.problem

    fINT, fBNC, fINC, minmaxx, minmaxt = init(solver, problem)
    x, t, y = diffusion_solve(J, minmaxx, dt, minmaxt, fINT, fBNC, fINC)

    # Plotting
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(5, 5, figsize=(10, 8))
   
    # Loop through each axis and data pair
   

    def animate_and_save(x, y, taus, filename="animation.gif"):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2, label="W(x, t)")
    
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_xlabel("x")
        ax.set_ylabel("W(x, t)")
        ax.legend()
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
        def init():
            line.set_data([], [])
            time_text.set_text("")
            return line, time_text
    
        def update(tau):
            line.set_data(x, y[:, tau])
            time_text.set_text(f"Time = {tau}")
            return line, time_text
    
        # Create the animation
        anim = FuncAnimation(fig, update, frames=taus, init_func=init, blit=True, interval=100)
    
        # Save the animation as a GIF
        anim.save(filename, writer=PillowWriter(fps=10))
        print(f"Animation saved as {filename}")
    
    # Example usage:
    taus = [i for i in range(50, 5000, 50)]
    animate_and_save(x, y, taus, filename="fokker_planck.gif")

   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t2d, x2d = np.meshgrid(t, x)
    ax.plot_surface(x2d, t2d, y, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('W(x, t)')
    
    plt.figure()
    plt.plot(x, y[:, -1], label='Final time')
    plt.xlabel('x')
    plt.ylabel('W(x, t)')
    plt.axvline(0.267044, color='black', linewidth=1, linestyle='--')
    plt.axvline(0.460957, color='black', linewidth=1, linestyle='--')
    plt.axvline(0.772358, color='black', linewidth=1, linestyle='--')
    plt.legend()
    plt.show()

    # Additional plots:
    '''
    plt.figure()
    plt.plot(x, y[:, -1], label='Final time')
    def U(x):
        a0, a1, a2, a3, a4, k, T = 300, -0.38, 1.37, -2, 1, 1, 1
        return a0 * k * T * (a4*x**4 + a3*x**3 + a2*x**2 + a1*x)
    potential = U(x)
    plt.plot(x, potential*1, label='Potential') #scaled just to see pattern
    plt.axvline(0.267044, color='black', linewidth=1, linestyle='--')
    plt.axvline(0.460957, color='black', linewidth=1, linestyle='--')
    plt.axvline(0.772358, color='black', linewidth=1, linestyle='--')
    plt.xlabel('x')
    plt.ylabel('W(x, t)')
    plt.legend()
    plt.show()
    '''
    
import sys
sys.argv = ["fokker_planck", "100", "0.00001", "implicit", "normal"]


# Entry point
if __name__ == "__main__":
    main()

    
    

