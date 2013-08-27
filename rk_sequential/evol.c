// (c) 2009 Frank Herrmann

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <time.h>

#include "ode45.h"

// post Newtonian RHS computation (see file pn_rhs.c)
void
rhs(// BH masses
    const double m1,
    // BH spins
    const double chi1, const double chi2,
    // state: note all vectors are unit vectors
    const double omega,
    const double S1ux, const double S1uy, const double S1uz,
    const double S2ux, const double S2uy, const double S2uz,
    const double LNx,  const double LNy,  const double LNz,
    // rhs
    double *domega,
    double *dS1ux, double *dS1uy, double *dS1uz,
    double *dS2ux, double *dS2uy, double *dS2uz,
    double *dLNx,  double *dLNy,  double *dLNz);

// -----------------------------------
// call the rhs function, the ode integrator expects this
// form for the functions, i.e. the state is stored in arrays,
// but the rhs functions uses indidvidual variables, so this
// just wraps
// state={x,y,z, Px,Py,Pz}
// rhsd={dx/dt,dy/dt,dz/dt, dPx/dt,dPy/dt,dPz/dt}
int 
rhs_wrapper (double *state, double *rhsd) {
	//printf("call at t=%f\n",time);
	const double m1 = 0.5;
	const double chi1 = 0.5,chi2 = 0.5;
	double omega = state[0];
	double S1ux = state[1], S1uy = state[2], S1uz = state[3];
	double S2ux = state[4], S2uy = state[5], S2uz = state[6];
	double LNx = state[7], LNy = state[8], LNz = state[9];

	rhs(m1, chi1, chi2,
		omega, S1ux, S1uy, S1uz, S2ux, S2uy, S2uz,
	    LNx, LNy, LNz,
	    &omega, &S1ux, &S1uy, &S1uz, &S2ux, &S2uy, &S2uz,
	    &LNx, &LNy, &LNz);
	// check for NaN's - this can happen if time-step is too large
	if (isnan(omega) ||
	    isnan(S1ux) || isnan(S1uy) || isnan(S1uz) ||
	    isnan(S2ux) || isnan(S2uy) || isnan(S2uz) ||
	    isnan(LNx) || isnan(LNy) || isnan(LNz)) {
		fprintf(stderr,
  "found NaN: dS1u=[%g,%g,%g] dS2u=[%g,%g,%g] dLN=[%g,%g,%g]\n",
	S1ux, S1uy, S1uz, S2ux, S2uy, S2uz, LNx, LNy, LNz);
		assert(0);
	}

	rhsd[0] = omega;
	rhsd[1] = S1ux; rhsd[2] = S1uy; rhsd[3] = S1uz;
	rhsd[4] = S2ux; rhsd[5] = S2uy; rhsd[6] = S2uz;
	rhsd[7] = LNx;  rhsd[8] = LNy;  rhsd[9] = LNz;

	return 0;
}


// main evolution function
int
main(int argc, char** argv) {
	// ODE evolution
	double omega = 0.004;
	double S1ux = 0.7071067811865476,S1uy = 0.7071067811865476,S1uz = 0;
	double S2ux = 0,S2uy = 0.7071067811865476,S2uz = 0.7071067811865476;
	double LNx = 0, LNy = 0, LNz = 1;
	const int N = 10; // number of variables: y={x,y,z,Px,Py,Pz}
	// time range. time integrator stops automatically, t2 is just large
	double t1 = 0, t2 = 1e10; // should be a high value, like 1e10;
	const double tol = 1e-7; // tolerance for solver
	// 1e-8 gives slightly better accuracy,
	// 1e-6 gives something completely different (try it out!)
	const double hmax = 10; // max time step (larger gives NaNs)

	int iters = 0, i = 0;
	if (argc > 1) {
		iters = atoi(argv[1]);
	}

	// initial state  
	double* ya = NULL;
	ya = (double*) malloc(iters*N*sizeof(double));
	for (i = 0; i < iters; i++) {
		ya[i*N + 0] = omega;
		ya[i*N + 1] = S1ux;
		ya[i*N + 2] = S1uy;
		ya[i*N + 3] = S1uz;
		ya[i*N + 4] = S2ux;
		ya[i*N + 5] = S2uy;
		ya[i*N + 6] = S2uz;
		ya[i*N + 7] = LNx;
		ya[i*N + 8] = LNy;
		ya[i*N + 9] = LNz;
	}

	time_t start = time(NULL);

#pragma omp parallel for
	for (i = 0; i < iters; i++) {
		// time integrator
		ode45(t1, t2, &ya[i*N], N,
		      tol, hmax,
		      rhs_wrapper);
	}

	time_t end = time(NULL);

	printf("Time to evolve %d particles: %ld secs\n", 
			iters, (int) end - start);

	for (i = 0; i < iters; i++) {
		printf("%d: dS1u=[%g,%g,%g] dS2u=[%g,%g,%g] dLN=[%g,%g,%g]\n",
		       i, ya[i*N + 1], ya[i*N + 2], ya[i*N + 3],
		       ya[i*N + 4], ya[i*N + 5], ya[i*N + 6],
		       ya[i*N + 7], ya[i*N + 8], ya[i*N + 9]);
	}

	free(ya);

	return 0;
}
