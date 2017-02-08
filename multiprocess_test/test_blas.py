__author__ = 'Alex'

import theano
import sys
import os
import numpy
import subprocess
import time


def execute(execute=True, verbose=True, M=2000, N=2000, K=2000,
            iters=10, order='C'):
    """
    :param execute: If True, execute a Theano function that should call gemm.
    :param verbose: If True, will print some Theano flags and env variables.
    :param M,N,K: The M,N,K size used by gemm.
    :param iters: The number of calls to gemm to do.

    :return: a tuple (execution time,
                      str that represents the implementation used)
    """

    if verbose:
        print 'Some Theano flags:'
        print '    blas.ldflags=', theano.config.blas.ldflags
        print '    compiledir=', theano.config.compiledir
        print '    floatX=', theano.config.floatX
        print '    device=', theano.config.device
        print 'Some OS information:'
        print '    sys.platform=', sys.platform
        print '    sys.version=', sys.version
        print '    sys.prefix=', sys.prefix
        print 'Some environment variables:'
        print '    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS')
        print '    OMP_NUM_THREADS=', os.getenv('OMP_NUM_THREADS')
        print '    GOTO_NUM_THREADS=', os.getenv('GOTO_NUM_THREADS')
        print
        print ('Numpy config: (used when the Theano flag'
               ' "blas.ldflags" is empty)')
        numpy.show_config()
        print 'Numpy dot module:', numpy.dot.__module__
        print 'Numpy location:', numpy.__file__
        print 'Numpy version:', numpy.__version__
        exit()
        if (theano.config.device.startswith("gpu") or
                theano.config.init_gpu_device.startswith("gpu")):
            print 'nvcc version:'
            subprocess.call((theano.sandbox.cuda.nvcc_compiler.nvcc_path,
                             "--version"))
            print

    a = theano.shared(numpy.ones((M, N), dtype=theano.config.floatX,
                                 order=order))
    b = theano.shared(numpy.ones((N, K), dtype=theano.config.floatX,
                                 order=order))
    c = theano.shared(numpy.ones((M, K), dtype=theano.config.floatX,
                                 order=order))
    f = theano.function([], updates=[(c, 0.4 * c + .8 * T.dot(a, b))])

    if any([x.op.__class__.__name__ == 'Gemm' for x in
            f.maker.fgraph.toposort()]):
        c_impl = [hasattr(thunk, 'cthunk')
                  for node, thunk in zip(f.fn.nodes, f.fn.thunks)
                  if node.op.__class__.__name__ == "Gemm"]
        assert len(c_impl) == 1
        if c_impl[0]:
            impl = 'CPU (with direct Theano binding to blas)'
        else:
            impl = 'CPU (without direct Theano binding to blas but with numpy/scipy binding to blas)'
    elif any([x.op.__class__.__name__ == 'GpuGemm' for x in
              f.maker.fgraph.toposort()]):
        impl = 'GPU'
    else:
        impl = 'ERROR, unable to tell if Theano used the cpu or the gpu:\n'
        impl += str(f.maker.fgraph.toposort())

    t0 = 0
    t1 = -1

    if execute:
        sync = (hasattr(theano, "sandbox") and
                hasattr(theano.sandbox, "cuda") and
                theano.sandbox.cuda.cuda_available)
        t0 = time.time()
        for i in range(iters):
            f()
        if sync:
            theano.sandbox.cuda.synchronize()
        t1 = time.time()
    return t1 - t0, impl

if __name__ =='__main__':
    execute(execute=False)