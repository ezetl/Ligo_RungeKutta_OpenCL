#define FLOAT float
#include "omega_rhs.inc"

/*
 * Calculates Runge-Kutta step.
 * ytmp: temporal array for intermediate results in ode45
 * y: initial array containing the initial state
 * k: array containing rhs temporaries k
 * a: array containing some constants TODO: que son las constantes?
 * nstep: number of step, used to acces k and a array. When working with b4, b5, this might be 1, because there is only one row, though it is the 7th and 8th step (TODO: check translation)
 * steps: number of total steps used. Useful to calculate the offset of array a. 
 * h: TODO: algo de los pasos
 */
__kernel void rk_step(__global FLOAT * ytmp,
                      __global FLOAT * y,
                      __global FLOAT * k,
                      __global FLOAT * a,
                      const int nstep,
                      const int steps,
                      const int nvars,
                      FLOAT h)
{
    int i=0;
    unsigned int id = get_global_id(0);


    for(i=0; i<nstep){
        /*adds 1 to i because in a[0] row, there are only zeros*/
        /*this is basically ytmp[i] = y[i] + h*a21*k1, etc...*/
        /*TODO: agregar chequeos sobre los indices, por ej: que el indice de a no overflowee de la matriz de datos de a*/
        ytmp[id] +=  a[nstep*steps+i] * k[nvars*i+id];
        /*k[number_variables offset by number of previous step (less than the actual number of step) plus the actual position in the array, that is the global id]*/

    }
    ytmp[id] *= h;
    ytmp[id] += y[id];
}


/* 
 * Calculates RHS and save it in k, with offset given by index i and the current number
 * of variables nvars.
 * state
 * rhsd: array for results.
 * nvars, steps: number of variables and steps of the system. Useful for 
 *               calculate offset in k.
 * curr_step: index used to calculate offset of k.
*/
__kernel void f_rhs(__global FLOAT * state,
                    __global FLOAT * rhsd,
                    const int nvars,
                    const int steps,
                    const int curr_step,
                    int error)
{
        /*TODO: parece complicado paralelizar eso. Basicamente cada una de 
         *las variables se calcula con calculos simples en 'rhs'. Hacer un
         *kernel para cada una no tienen sentido, hay que pasarle muchas */
//        const int index = get_global_id(0);
        /*TODO: ahora las masas son parametros, modificarlo luego*/
        const FLOAT m1 = 0.5;
        const FLOAT chi1 = 0.5,chi2 = 0.5;
        FLOAT omega = state[0];
        FLOAT S1ux  = state[1],
              S1uy  = state[2],
              S1uz  = state[3];
        FLOAT S2ux  = state[4],
              S2uy  = state[5],
              S2uz  = state[6];
        FLOAT LNx   = state[7],
              LNy   = state[8],
              LNz   = state[9];
        
        rhs(m1, chi1, chi2, omega, 
            S1ux, S1uy, S1uz,
            S2ux, S2uy, S2uz,
            LNx, LNy, LNz,
            &omega,
            &S1ux, &S1uy, &S1uz,
            &S2ux, &S2uy, &S2uz,
            &LNx, &LNy, &LNz);

        // check for NaN's - this can happen if time-step is too large
        if (isnan(omega) ||
            isnan(S1ux) || isnan(S1uy) || isnan(S1uz) ||
            isnan(S2ux) || isnan(S2uy) || isnan(S2uz) ||
            isnan(LNx) || isnan(LNy) || isnan(LNz)) {
            error += 1;
        }

        rhsd[nvars*curr_step + 0] = omega;
        rhsd[nvars*curr_step + 1] = S1ux;
        rhsd[nvars*curr_step + 2] = S1uy;
        rhsd[nvars*curr_step + 3] = S1uz;
        rhsd[nvars*curr_step + 4] = S2ux;
        rhsd[nvars*curr_step + 5] = S2uy;
        rhsd[nvars*curr_step + 6] = S2uz;
        rhsd[nvars*curr_step + 7] = LNx;
        rhsd[nvars*curr_step + 8] = LNy;
        rhsd[nvars*curr_step + 9] = LNz;
}


