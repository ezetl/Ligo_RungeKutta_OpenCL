/** (c) 2009 Frank Herrmann
 * matlab ode45 integrator
 * WARN: single precision accuracy
 * 		 don't forget to change powf() to pow() if switching to double
 **/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "ode45.h"

/**
 * allocate memory for array. only used locally.
 * return pointer to float array
 * N: number of elements in array
 * name: name of array, just for diagnostics printing
 **/
static double *
alloc (int N, char *name) {
	assert(N > 0);
	assert(name != NULL);
	double *ptr = (double *) malloc(N*sizeof(double));
	if (ptr == NULL) {
		fprintf(stderr,"WARN: allocation of array '%s' failed\n", name);
		assert(0);
	}
	return ptr;
}


void print_array(const double * arr, const int n){
    printf("[ ");
    for (int i=0; i<n-1; i++){
        printf("%lf, ", arr[i]);
    }
    printf("%lf ]\n", arr[n-1]);
}

/**
 * ode45: matlab ode integrator
 * t1: initial time
 * t2: final time
 * y[N]: initial state (array gets overwritten)
 * N: number of variables, i.e. size of y
 * tol: tolerance for solver
 * hmax: maximum step-size
 * Nout: # intermediate steps to store (can be 0)
 * tout: array for intermediate time values
 * yout: array for intermediate state values
 * 		 needs to be size yout[Nout][Nvars]
 * f_rhs(): function which calls rhs
 * 			returns 0 if everything is OK and !=0 if something is wrong
 **/
void
ode45(double t1, double t2, double *y, int N,
      double tol, double hmax,
      int (*f_rhs)(double *state, double *rhs)) {
	// minimum time step
	const double hmin=(t2-t1)/1e20;

	// dormand-price coefficients
	const double a21 = 1./5.,
		a31 = 3./40., a32 = 9./40.,
		a41 = 44./45., a42 = -56./15., a43 = 32./9.,
		a51 = 19372./6561., a52 = -25360./2187., a53 = 64448./6561.,
			a54 = -212./729.,
		a61 = 9017./3168., a62 = -355./33., a63 = 46732./5247., 
			a64 = 49./176., a65 = -5103./18656.,
		a71 = 35./384., a73 = 500./1113., a74 = 125./192.,
			a75 = -2187./6784., a76 = 11./84.; // note: a72=0
	
	// 4th order b-coeffs: b4_2=0
	const double b4_1 = 5179./57600., b4_3 = 7571./16695., 
		b4_4 = 393./640., b4_5 = -92097./339200.,
		b4_6 = 187./2100., b4_7 = 1./40.;

	// 5th order b-coeffs: b5_2=0 & b5_7=0
	const double b5_1 = 35./384., b5_3 = 500./1113., b5_4 = 125./192.,
		b5_5 = -2187./6784., b5_6 = 11./84.;

	double *k1, *k2, *k3, *k4, *k5, *k6, *ytmp, *y4, *y5, *k7;

	double time = t1;
	double h = (t2-t1)/100.; // try 100 steps by default
	double delta = -1e10, yinf = -1e10, dif, ya, tau;

	const double power=1./6.;

	int nstp = 0, n_ok = 0, n_bad = 0, ierr = 0, i = 0, diff = 0;

	assert(y != NULL);
	assert(t2 > t1);
	assert(tol > 0);

	h = (h > hmax) ? hmax : h;
	h = (h < hmin) ? hmin : h;

	// allocate rhs temporaries ki
	k1 = alloc(N,"k1"); k2 = alloc(N,"k2"); k3 = alloc(N,"k3");
	k4 = alloc(N,"k4"); k5 = alloc(N,"k5"); k6 = alloc(N,"k6");
	ytmp = alloc(N,"ytmp"); y4 = alloc(N,"y4"); y5 = alloc(N,"y5");
	k7 = k2; // recycle k2 memory

	while (1) {
		assert(h >= 0);
		if (time >= t2 || h < hmin || ierr != 0) break;
		/*if (time+h > t2) h = t2-time;*/
		h = (time+h > t2)*(t2-time) + h*(time+h <= t2);

		// compute slopes k1...k7
		ierr += f_rhs(y, k1);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+h*a21*k1[i];

/*			printf("ytmp\n");*/
/*			print_array(ytmp, N);*/
		ierr += f_rhs(ytmp, k2);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+h*(a31*k1[i]+a32*k2[i]);
/*			printf("ytmp\n");*/
/*			print_array(ytmp, N);*/


		ierr += f_rhs(ytmp, k3);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+h*(a41*k1[i]+a42*k2[i]+a43*k3[i]);
/*			printf("ytmp\n");*/
/*			print_array(ytmp, N);*/


		ierr += f_rhs(ytmp, k4);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]);
		ierr += f_rhs(ytmp, k5);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+
				h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]);
/*				printf("ytmp\n");*/
/*				print_array(ytmp, N);*/


		ierr += f_rhs(ytmp, k6);
		for (i = 0; i < N; i++)
			ytmp[i] = y[i]+
				h*(a71*k1[i]+ +a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]);


		ierr += f_rhs(ytmp, k2); // note that k2 is used for k7
		// 4th order estimate
		for (i = 0; i < N; i++)
			y4[i] = y[i]+h*(b4_1*k1[i]+b4_3*k3[i]+b4_4*k4[i]+
					b4_5*k5[i]+b4_6*k6[i]+b4_7*k7[i]);
		// 5th order estimate
		for (i = 0; i < N; i++)
			y5[i] = y[i]+h*(b5_1*k1[i]+b5_3*k3[i]+b5_4*k4[i]+
					b5_5*k5[i]+b5_6*k6[i]);
		
		// compare truncation error und acceptable error
		delta = -1e10;
		yinf = -1e10;
		for (i = 0; i < N; i++) {
			dif = fabs(y5[i]-y4[i]);
			
			/*if (dif > delta) delta = dif;*/
			delta = dif*(dif > delta) + delta*(dif <= delta);
			
			ya = fabs(y[i]);
			
			/*if (ya > yinf) yinf = ya;*/
			yinf = ya*(ya > yinf) +  yinf*(ya <= yinf);
		}
		tau = fmax(yinf, 1)*tol;
		
		// optimizing this frequent condition out
		diff = y5[0] <= 0.1+tol;

		time = time+h*(delta <= tau && diff); // use 5th order estimate for y
		/*Asigno nuevos valores a y[i]*/
		for (i = 0; i < N; i++) 
		    y[i] = y5[i]*(delta <= tau && diff) + y[i]*(1-(delta<=tau && diff));
		n_ok = n_ok + (delta <= tau && diff);
		//} else { // non-acceptable error, reject step
		n_bad = n_bad + (1 - (delta <= tau && diff));
			//		}
		
		// adjust time step
		/*if (delta == 0) delta = 1e-16;*/
		delta = 1e-16*(delta == 0) + delta;

        if(nstp==1){
            break;
        }

		// |omega - omega_final| < tolerance
		if (fabs(y5[0]-0.1) < tol){
            printf("pasos: %d\n", nstp);
            printf("y5[0] = %f", y5[0]);		
			break;
		}
/*	    if(nstp==3){*/
/*	        break;*/
/*		}*/
		nstp++;

		//h=fmin(hmax,0.8*h*powf(tau/delta,power)); single precision
		h = fmin(hmax, 0.8*h*pow(tau/delta, power))*(h != 0)*diff+ h*(1-diff)/2;
	}
	printf("ntsp = %d\n", nstp);
}

