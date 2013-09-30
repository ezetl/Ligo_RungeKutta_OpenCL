#define FLOAT int

__kernel void test(__global FLOAT * a,
                   __global FLOAT * h,
                   __global FLOAT * res)
{
    unsigned int id = get_global_id(0);
/*    unsigned int hid = get_group_id(0);*/
    unsigned int hid = get_group_id(0);
    res[id] = a[id] + h[hid];

}

