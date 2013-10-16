#define FLOAT float
#define DELTA (-1e10)
#include "omega_rhs.inc"


__kernel void check_step(__global FLOAT * h,
	                     __global FLOAT * time,
	                     __global int * stops,
	                     const FLOAT t2,
	                     const int hmin)
{
    const unsigned int id = get_group_id(0);
    
    if (time[id] >= t2 || h[id] < hmin || stops[id] == 1) {
        stops[id] = 1;
    }
    /*if (time+h > t2) h = t2-time;*/
    h[id] = (t2-time[id])*(time[id]+h[id] > t2) + h[id]*(time[id]+h[id] <= t2);
}


__kernel void evaluate_step(__global FLOAT * y,
                       __global FLOAT * y4,
                       __global FLOAT * y5,
                       __global FLOAT * tau,
                       __global FLOAT * delta,
                       const FLOAT tol,
                       const int nvars)
{
    /*TODO: esto se hace una vez por system, ver como*/
    unsigned int gid = get_group_id(0);
    unsigned int id = get_global_id(0);
    /*offset to the first variable of each state (omega)*/
    unsigned int om_offs = nvars * gid;
    FLOAT yinf = -1e10;
    FLOAT dif = 0;
    FLOAT abs_y = 0;
    int i=0;

    delta[gid] = DELTA;

    for(i=0; i<nvars; i++){
        dif = fabs(y5[om_offs + i] - y4[om_offs + i]);
        delta[gid] = dif * (dif > delta[gid]) + delta[gid] * (dif <= delta[gid]);
        abs_y = fabs(y[om_offs + i]);
        yinf = abs_y * (abs_y > yinf) + yinf * (abs_y <= yinf);
    }

    tau[gid] = fmax(yinf, 1) * tol;
}


__kernel void update_variables(__global FLOAT * y5,
                          __global FLOAT * delta,
                          __global FLOAT * tau,
                          __global FLOAT * time,
                          __global FLOAT * h,
                          __global FLOAT * y,
                          __global FLOAT * n_ok,
                          __global FLOAT * n_bad,
                          __global int * stop,
                          const FLOAT tol,
                          const FLOAT hmax,
                          const FLOAT final_omega,
                          const int nvars
                          )
{
    unsigned int id = get_global_id(0);
    unsigned int gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    /*Group offset, for indexing y5[0] = omega*/
    unsigned int om_offs = gid * nvars;

    const FLOAT power = 1./(FLOAT)6.;
    int ome_cond = (y5[om_offs] <= final_omega + tol);
    int diff = (delta[gid] <= tau[gid]) && ome_cond;

    time[gid] = time[gid] + h[gid] * diff;

    /*New values of y[i]*/
    y[id] = y5[id] * diff + y[id] * (1 - diff);

    n_ok[gid] = n_ok[gid] + diff;
    n_bad[gid] = n_bad[gid] + (1 - diff);

    delta[gid] = 1e-16 * (delta[gid] == 0) + delta[gid];

    if(fabs(y5[om_offs] - final_omega) < tol){
        stop[gid] = 1;
    }

    /*XXX: ojo, el 2 lo pone como un entero, es asi?*/
    h[gid] = fmin(hmax, 0.8f * h[gid] * pow(tau[gid] / delta[gid], power)) \
             * (h[gid] != 0) * ome_cond + h[gid] * (1 - ome_cond) / 2;
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
                      const __global FLOAT * k,
                      const __global FLOAT * a,
                      const __global FLOAT * h,
                      const int nstep,
                      const int steps,
                      const int nvars)
{
    unsigned int i=0;
    unsigned int id = get_global_id(0);
    unsigned int gid = get_group_id(0);
    unsigned int lid = get_local_id(0);

    FLOAT tmp = ytmp[id];

    for(i=0; i<nstep; i++){
        /*adds 1 to i because in a[0] row, there are only zeros*/
        /*this is basically ytmp[i] = y[i] + h[]*a21*k1, etc...*/
        /*Note: nstep is never zero. That is because row 0 of array "a" is never used*/
        /*TODO: agregar chequeos sobre los indices, por ej: que el indice de a no overflowee de la matriz de datos de a*/
        /*Use steps-1, arrays a, b4, and b5 are bounded by 6 (STEPS-1)*/
        //ytmp[id] +=  a[nstep*(steps-1)+i] * k[i*nvars + hid*steps + lid];
        //k[ offset of current step + offset of current batch + offset of variable ]
        tmp +=  a[nstep*(steps-1) + i] * k[i*nvars + gid*steps*nvars + lid];
    }

    tmp *= h[gid];
    tmp += y[id];
    ytmp[id] = tmp;
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
                    __global FLOAT * error,
                    const int nvars,
                    const int steps,
                    const int curr_step)
{
    /*TODO: la idea es que esto se haga una vez por work_group, ver como hacerlo*/
        /*TODO: ahora las masas son parametros, modificarlo luego*/
        const FLOAT m1 = 0.5;
        const FLOAT chi1 = 0.5, chi2 = 0.5;
        unsigned int gid = get_group_id(0);
        unsigned int id = get_global_id(0);
        unsigned int lid = get_local_id(0);
        
        FLOAT omega = state[gid];
        FLOAT S1ux  = state[gid + 1],
              S1uy  = state[gid + 2],
              S1uz  = state[gid + 3];
        FLOAT S2ux  = state[gid + 4],
              S2uy  = state[gid + 5],
              S2uz  = state[gid + 6];
        FLOAT LNx   = state[gid + 7],
              LNy   = state[gid + 8],
              LNz   = state[gid + 9];

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
                error[gid] = 1;
        }
//        if (isnan(omega) ) {
//                error[gid] = 100;
//        }
//        if (isnan(S1ux) ) {
//                error[gid] = 200;
//        }
//        if (isnan(S1uy) ) {
//                error[gid] = 300;
//        }
//        if (isnan(S1uz) ) {
//                error[gid] = 400;
//        }
//        if (isnan(S2ux) ) {
//                error[gid] = 500;
//        }
//        if (isnan(S2uy) ) {
//                error[gid] = 600;
//        }
//        if (isnan(S2uz)) {
//                error[gid] = 700;
//        }
//        if (isnan(LNx)) {
//                error[gid] = 800;
//        }
//        if (isnan(LNy)) {
//                error[gid] = 900;
//        }
//        if (isnan(LNz)) {
//                error[gid] = 1000;
//        }

        unsigned int offs = curr_step*nvars + gid*steps*nvars;
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
//        rhsd[offs] = state[gid];
//        rhsd[offs + 1] = state[gid + 1];
//        rhsd[offs + 2] = state[gid + 2];
//        rhsd[offs + 3] = state[gid + 3];
//        rhsd[offs + 4] = state[gid + 4];
//        rhsd[offs + 5] = state[gid + 5];
//        rhsd[offs + 6] = state[gid + 6];
//        rhsd[offs + 7] = state[gid + 7];
//        rhsd[offs + 8] = state[gid + 8];
//        rhsd[offs + 9] = state[gid + 9];
}
