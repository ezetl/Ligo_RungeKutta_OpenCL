// (c) 2009 Frank Herrmann

// ode45: matlab ode integrator
// t1: initial time
// t2: final time
// y[N]: initial state (array gets overwritten)
// N: number of variables, i.e. size of y
// tol: tolerance for solver
// hmax: maximum step-size
// Nout: number of intermediate steps to store, if 0 then no steps are stored
// tout: array which will hold the intermediate time values
// yout: array which holds the intermediate state values, needs to be of size yout[Nout][Nvars]
// f_rhs(): function which calls rhs
// note f_rhs returns 0 if everything is OK and !=0 if something is wrong
void
ode45(double t1, double t2, double *y, int N,
      double tol, double hmax,
      int (*f_rhs)(double *state, double *rhs));

#define OMEPS 1e-7
