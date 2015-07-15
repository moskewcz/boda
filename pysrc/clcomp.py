import pycl as cl
import os
import sys

def run_mxplusb( prog, queue ):
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

def run_conv( prog, queue ):
    func = prog['conv__num_imgs_20__in_pad_3__in_dim_0_227__in_dim_1_227__conv_has_relu_1__kern_sz_7__stride_2__out_chans_64__in_chans_3']
    print func
    func.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem)
    in_ar = cl.array('f', range(100 * 1000 * 1000))
    in_buf, in_evt = cl.buffer_from_pyarray(queue, in_ar, blocking=False)
    filt_buf = in_buf.empty_like_this()
    bias_buf = in_buf.empty_like_this()
    out_buf = in_buf.empty_like_this()
    #run_evt = func(filt_buf, bias_buf, in_buf, out_buf).on(queue, gsize=(63,), lsize=(32,), wait_for=in_evt)
    func.setarg( 0, filt_buf )
    func.setarg( 1, bias_buf )
    func.setarg( 2, in_buf )
    func.setarg( 3, out_buf )
    run_evt = cl.clEnqueueNDRangeKernel( queue, func, gsize=(120*2166,), lsize=(120,), wait_for=in_evt)

    out, evt = cl.buffer_to_pyarray(queue, out_buf, wait_for=run_evt, like=in_ar)
    print "start wait"
    evt.wait()
    print "end wait"
    print out[0:10]


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
    #run_mxplusb( prog, queue )
    run_conv( prog, queue )


import argparse
parser = argparse.ArgumentParser(description='command-line OpenCL compiler.')
parser.add_argument('--src-fn', metavar="FN", type=str, default="../../test/conv1.cl", help="OpenCL source filename" )
args = parser.parse_args()
ocl_src = open(args.src_fn).read()
ocl_init( ocl_src )

