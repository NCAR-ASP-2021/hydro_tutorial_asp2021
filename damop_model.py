"""
Applying an impact model for hydroelectric dam management driven by
a time series of runoff data

Author: 2020, John Methven
"""
#
# Library functions needed to run damop_model()
#
from scipy import optimize
from scipy import signal

def damop_model(runoffarr, dt, hdam, catcharea, kappa, wmax, wmin, mu, sigma):
    '''
    Implementation of the dam operation model of Hirsch et al (2014).
    Input: 
    :runoffarr  - input time series for runoff data
    :dt         - runoff accumulation interval per record
    :hdam       - dam height used also as maximum water head
    :catcharea  - catchment area for the dam
    :kappa      - parameter relating reservoir depth to volume
    :wmax       - maximum flow rate through turbines
    :wmin       - minimum flow rate to maintain some power generation
    :mu         - conversion factor to obtain power in units of MW
    :sigma      - operational efficiency of power generation by dam
    Output: 
    :inflow     - input time series for inflow to reservoir  
    :x          - output time series for water head at dam
    :w          - output solution for optimum flow rate through turbines
    :gout       - value of time integrated generation for optimum solution (MW-days)
    '''       
    print()
    print('damop_model has been called')
    print('wmax = ',wmax,'   wmin = ',wmin)
    #
    # Convert runoff data from units of m to an equivalent flow in m^3 s^-1
    # Assume that the same runoff rate applies to the entire catchment area for dam
    #
    hmax = 0.5*hdam
    hmin = 0.2*hmax
    #
    # Set parameter used to control computational mode using filter similar to Robert-Asselin
    #
    alpha = 0.1
    #
    runoffave = np.mean(runoffarr)
    #runoffarr[:] = runoffave
    inflow = catcharea*runoffarr/dt
    inmax = max(inflow)
    n = len(inflow)
    #
    # The dam management optimization model is set up in the mathematical form of a 
    # quadratic programming problem.
    # The only input time series is the inflow to the reservoir.
    # The model solves for the water head at the dam maximizing power generation.
    # This then gives the flow rate through the turbines.
    # However, contraints are applied on maximum and minimum water level 
    # and maximum/minimum flow rate through the turbines.
    #
    # The equation for generation can be written in the form
    # 
    # G = 0.5*H^T P H + q^T H
    #
    # where H is the head time series we are solving for (a 1-D array) and 
    # P is a matrix and q is also a 1-D time series (scaled inflow).
    # The notation ^T means the transpose of the matrix. 
    # Quadratic programming aims to minimize -G which is equivalent to max(G).
    #
    q = -mu*sigma*inflow
    pmat = np.zeros((n, n))
    cmat = np.zeros((n, n))
    umat = np.zeros((n, n))
    for i in range(n-1):
        pmat[i, i] = -1
        pmat[i, i+1] = 1
        umat[i, i] = 1
    umat[n-1, n-1] = 1 
    for j in range(n-2):
        i = j+1
        cmat[i, i-1] = -1 + 0.5*alpha
        cmat[i, i]   = -alpha
        cmat[i, i+1] = 1 + 0.5*alpha
    
    pscal = 2*mu*sigma*(kappa/dt)*cmat
    gscal = -(kappa/dt)*cmat
    wmaxcons = np.zeros(n)
    wmincons = np.zeros(n)
    wmaxcons[:] = -inflow[:]+wmax
    wmincons[:] = -inmax+wmin
    hscal = umat
    hmaxcons = np.ones(n)*hmax
    hmincons = np.ones(n)*hmin
    
    vmat = np.concatenate((gscal, -gscal, hscal, -hscal), axis=0)
    vcons = np.concatenate((wmaxcons, -wmincons, hmaxcons, -hmincons))
    
    print('Now apply quadratic minimization technique')
    
    def gen(x, sign=1.):
        return sign * (0.5*np.dot(x.T, np.dot(pscal, x)) + np.dot(q.T, x))
    
    def jac(x, sign=1.):
        return sign * (np.dot(x.T, pscal) + q.T)
    
    cons = {'type':'ineq',
            'fun':lambda x: vcons - np.dot(vmat, x),
            'jac':lambda x: -vmat}
    
    opt = {'disp':True, 'maxiter':100, 'ftol':1e-08}

    #
    # Obtain solution by minimization nouter times and average the results
    # to remove noise.
    # Note that the minimize method does not always find a solution consistent 
    # with the contraints imposed (depending on the first guess data) and these
    # failed attempts are not included in the average solution.
    #
    nouter = 5
    istsuccess = 1
    ic = -1
    for io in range(nouter):
    #while istsuccess == 1:
        #
        # First guess values for x (water head).
        # Random variation on top of constant level.
        # Smooth to reduce 2-grid noise in input data.
        #
        ic = ic+1
        xinit = hmax*(0.1+0.01*np.random.randn(n))
        xinit = signal.medfilt(xinit, kernel_size=5)
        
        res_cons = optimize.minimize(gen, xinit, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)
        xup = res_cons['x']
        fup = res_cons['fun']  
        stexit = res_cons['status']
    
        if stexit != 4:
            if istsuccess == 1:
                x = xup
                x = signal.medfilt(x, kernel_size=5)
                f = fup
                print('Constrained optimization')
                print(res_cons)
                print('iter ',ic,' f = ',f)
                istsuccess = 0
            else:
                if (fup/f) < 2:
                    afac = float(ic+1)/nouter
                    x = afac*x + (1-afac)*xup
                    x = signal.medfilt(x, kernel_size=5)
                    f = afac*f + (1-afac)*fup
                    print('iter ',ic,' f = ',f)
        if ic == nouter:
            print(nouter,' outer iterations finished without reaching result')
            istsuccess = 1
    # end outer loop
    
    #w = np.dot(gscal, x)
    w = np.dot(gscal, x) + inflow
    gout = -f
    
    return inflow, x, w, gout
