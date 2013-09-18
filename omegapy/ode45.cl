#define FLOAT float
#include "omega_rhs.inc"


__kernel void rk_step(__global FLOAT * b5,
                      __global FLOAT * b4,
                      __global FLOAT * a)
{
    unsigned int id = get_global_id(0);

    b5[id] += 100; 
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

        error = 34;

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


