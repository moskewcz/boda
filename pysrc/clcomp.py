import pycl as cl
import os
import sys


def ocl_init( ocl_src ):
    platforms = cl.clGetPlatformIDs()
    use_devices = None
    for platform in platforms:
        try:
            devices = cl.clGetDeviceIDs(platform,device_type=cl.CL_DEVICE_TYPE_GPU)
            use_devices = devices[0:1] # arbitraily choose first device
        except cl.DeviceNotFoundError:
            pass
        if use_devices is not None: break
    if use_devices is None: raise ValueError( "no GPU openCL device found" )
    assert use_devices is not None
    print( "OpenCL use_devices: " + str(use_devices) )

    context = cl.clCreateContext(use_devices)
    queue = cl.clCreateCommandQueue(context)

    prog = cl.clCreateProgramWithSource( context, ocl_src ).build()
    print prog
    func = prog['mxplusb']
    print func

    func.argtypes = (cl.cl_float, cl.cl_mem, cl.cl_float, cl.cl_mem)
    x = cl.array('f', range(100))
    x_buf, in_evt = cl.buffer_from_pyarray(queue, x, blocking=False)
    y_buf = x_buf.empty_like_this()
    run_evt = func(2, x_buf, 5, y_buf).on(queue, len(x), wait_for=in_evt)
    y, evt = cl.buffer_to_pyarray(queue, y_buf, wait_for=run_evt, like=x)
    evt.wait()
    print y[0:10]

import argparse
parser = argparse.ArgumentParser(description='command-line OpenCL compiler.')
parser.add_argument('--src-fn', metavar="FN", type=str, default="../../test/mxplusb.cl", help="OpenCL source filename" )
args = parser.parse_args()
ocl_src = open(args.src_fn).read()
ocl_init( ocl_src )

