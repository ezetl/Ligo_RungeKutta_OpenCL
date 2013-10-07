        #TODO: chequear como se usaba el error en el rhs
#define FLOAT float
#include "omega_rhs.inc"

/* TODO: change parameters in a near future, when using batchs, to arrays
 */
void check_step(FLOAT * h, FLOAT * time, int * stops, const FLOAT t2, const int hmin)
{
    const int id = get_group_id(0);

    if (time[id] >= t2 || h[id] < hmin || stops[id] != 0) {
        stops[id] = 1;
    }
    /*if (time+h > t2) h = t2-time;*/
    h[id] = (t2-time[id])*(time[id]+h[id] > t2) + h[id]*(time[id]+h[id] <= t2);
}


__kernel update_variables()
{
}


/*
 * Calculates Runge-Kutta step.
 * ytmp: temporal array for intermediate results in ode45
 * y: initial array containing the initial state
 * k: array containing rhs temporaries k
 * a: array containing some constants TODO: que son las constantes?
 * nstep: number of step, used to acces k and a array. With arrays b4 and b5 this should be the number 7 and 6 respectively, because is the number of elements of those arrays.
 * steps: number of total steps that are going to be used, for example, array a has 7 steps. Useful to calculate the offset of array a. Array b4 and b5 should use number 0
 * h: TODO: algo de los pasos
 */
__kernel void rk_step(__global FLOAT * ytmp,
                      __global FLOAT * y,
                      __global FLOAT * k,
                      __global FLOAT * a,
                      __global FLOAT * h,
                      const int nstep,
                      const int steps,
                      const int nvars)
{
    int i=0;
    unsigned int id = get_global_id(0);
    unsigned int hid = get_group_id(0);
    unsigned int lid = get_local_id(0);

    for(i=0; i<nstep; i++){
        /*adds 1 to i because in a[0] row, there are only zeros*/
        /*this is basically ytmp[i] = y[i] + h[]*a21*k1, etc...*/
        /*Note: nstep is never zero. That is because row 0 of array "a" is never used*/
        /*TODO: agregar chequeos sobre los indices, por ej: que el indice de a no overflowee de la matriz de datos de a*/
        ytmp[id] +=  a[nstep*steps+i] * k[i*nvars + hid*steps + lid];

    }
    ytmp[id] *= h[hid];
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
 * stride:
*/
__kernel void f_rhs(__global FLOAT * state,
                    __global FLOAT * rhsd,
                    const int nvars,
                    const int steps,
                    const int curr_step,
                    __global FLOAT * error)
{
        /*TODO: parece complicado paralelizar eso. Basicamente cada una de 
         *las variables se calcula con calculos simples en 'rhs'. Hacer un
         *kernel para cada una no tienen sentido, hay que pasarle muchas */
//        const int index = get_global_id(0);
        /*TODO: ahora las masas son parametros, modificarlo luego*/
        const FLOAT m1 = 0.5;
        const FLOAT chi1 = 0.5,chi2 = 0.5;
        unsigned int gid = get_group_id(0);

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
            error[gid] += 1;
        }

        unsigned int offs = curr_step*nvars + gid*steps;
        rhsd[offs] = omega;
        rhsd[offs + 1] = S1ux;
        rhsd[offs + 2] = S1uy;
        rhsd[offs + 3] = S1uz;
        rhsd[offs + 4] = S2ux;
        rhsd[offs + 5] = S2uy;
        rhsd[offs + 6] = S2uz;
        rhsd[offs + 7] = LNx;
        rhsd[offs + 8] = LNy;
        rhsd[offs + 9] = LNz;
}


