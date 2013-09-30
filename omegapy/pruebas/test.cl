#define FLOAT float

__kernel void rk_step(__global FLOAT * a,
                      __global FLOAT * h,
                      const int batch,
                      const int nvars)
{
    unsigned int id = get_global_id(0);
    
    ytmp[id] *= h[];
    ytmp[id] += y[id];
}

