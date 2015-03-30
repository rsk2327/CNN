# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:50:57 2015

@author: roshan
"""

from theano import *

import numpy
import theano.tensor as T
from theano.tensor.signal import downsample
from theano import gof, Op, tensor, Variable, Apply


d = T.dmatrix('d')
w = T.dmatrix('w')
a = numpy.ndarray((4,4))
t = numpy.asarray([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]]])
e=numpy.asarray([[1,2,7,8],[3,4,1,3],[8,5,3,9],[3,2,5,12]])
e1 = numpy.asarray([[[[1,2,7,8],[3,4,1,3],[8,5,3,9],[3,2,5,12]]]])
print e








class DownsampleFactorMax(Op):
    """For N-dimensional tensors, consider that the last two
    dimensions span images.  This Op downsamples these images by a
    factor ds, by taking the max over non- overlapping rectangular
    regions.

    """
    __props__ = ('ds', 'ignore_border', 'st', 'padding')
				
    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        :param imgshape: the shape of a tensor of images. The last two elements
            are interpreted as the number of rows, and the number of cols.
        :type imgshape: tuple, list, or similar of integer or
            scalar Theano variable.

        :param ds: downsample factor over rows and columns
                   this parameter indicates the size of the pooling region
        :type ds: list or tuple of two ints

        :param st: the stride size. This is the distance between the pooling
                   regions. If it's set to None, in which case it equlas ds.
        :type st: list or tuple of two ints

        :param ignore_border: if ds doesn't divide imgshape, do we include an
            extra row/col of partial downsampling (False) or ignore it (True).
        :type ignore_border: bool

        :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.
        :type padding: tuple of two ints

        :rtype: list
        :returns: the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but with last
            two elements reduced as per the downsampling & ignore_border flags.
        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        #padding[0] is the padding added above and below the image. hence no of rows increases by 2*padding[0]
        r += padding[0] * 2
        #padding[1] is the padding added left and right the image. hence no of columns increases by 2*padding[1]
        c += padding[1] * 2

        if ignore_border:
            out_r = (r - ds[0]) // st[0] + 1
            out_c = (c - ds[1]) // st[1] + 1
            if isinstance(r, theano.Variable):
                nr = tensor.maximum(out_r, 0)
            else:
                nr = numpy.maximum(out_r, 0)
            if isinstance(c, theano.Variable):
                nc = tensor.maximum(out_c, 0)
            else:
                nc = numpy.maximum(out_c, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = tensor.switch(tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   tensor.maximum(0, (r - 1 - ds[0])
                                                  // st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1])
                                                  // st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0)):
        """
        :param ds: downsample factor over rows and column.
                   ds indicates the pool region size.
        :type ds: list or tuple of two ints

        :param ignore_border: if ds doesn't divide imgshape, do we include
            an extra row/col of partial downsampling (False) or
            ignore it (True).
        :type ignore_border: bool

        : param st: stride size, which is the number of shifts
            over rows/cols to get the the next pool region.
            if st is None, it is considered equal to ds
            (no overlap on pooling regions)
        : type st: list or tuple of two ints

        :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.
        :type padding: tuple of two ints

        """
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "DownsampleFactorMax downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')

    def __str__(self):
        return '%s{%s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.ds, self.st, self.ignore_border, self.padding)

# During Theano graph construction the Op creates an Apply node by calling the make_node() method
# Apply nodes are connected to Variable nodes and take care of the intermediate computations 
    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [x.type()])


    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMax requires 4D input for now')
        # z_shape is the shape of the tensor that the Op will out put(as determined by the out_shape method).
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        # If the output location (z[0]) is not of that shape determined by out_shape then set z[0] as new variable of required shape
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        
        #zz is the shape of the maxpooled tensor. The number of rows is given by the 2nd last element of zz's shape vector while the number of 
        # columns is givn by the last element of zz's shape vector  
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
# Carry out the padding process. The padded elements have the fill value. The new padded tensor is y. From here onwards y will be the input.
        if self.padding != (0, 0):
            fill = x.min()-1.
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype) + fill
            y[:, :, pad_h:(img_rows-pad_h), pad_w:(img_cols-pad_w)] = x
        else:
            y = x
            
        
        # max pooling
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = __builtin__.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = __builtin__.min(col_st + ds1, img_cols)
                        zz[n, k, r, c] = y[
                            n, k, row_st:row_end, col_st:col_end].max()

    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st, self.padding)
        return [shp]

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        maxout = self(x)
        return [DownsampleFactorMaxGrad(self.ds,
                                        ignore_border=self.ignore_border,
                                        st=self.st, padding=self.padding)(
                                            x, maxout, gz)]

    def c_code(self, node, name, inp, out, sub):
        # No implementation is currently for the case where
        # the stride size and the pooling size are different.
        # An exception is raised for such a case.
        if self.ds != self.st or self.padding != (0, 0):
            raise theano.gof.utils.MethodNotDefined()
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        return """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        z_shp0 = PyArray_DIMS(%(x)s)[2] / %(ds0)s;
        z_shp1 = PyArray_DIMS(%(x)s)[3] / %(ds1)s;
        if (%(ignore_border)s)
        {
            x_shp0_usable = z_shp0 * %(ds0)s;
            x_shp1_usable = z_shp1 * %(ds1)s;
        }
        else
        {
            z_shp0 += (PyArray_DIMS(%(x)s)[2] %% %(ds0)s) ? 1 : 0;
            z_shp1 += (PyArray_DIMS(%(x)s)[3] %% %(ds1)s) ? 1 : 0;
            x_shp0_usable = PyArray_DIMS(%(x)s)[2];
            x_shp1_usable = PyArray_DIMS(%(x)s)[3];
        }
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_shp0)
          ||(PyArray_DIMS(%(z)s)[3] != z_shp1)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        if (z_shp0 && z_shp1)
        {
            for(int b=0;b<PyArray_DIMS(%(x)s)[0];b++){
              for(int k=0;k<PyArray_DIMS(%(x)s)[1];k++){
                int mini_i = 0;
                int zi = 0;
                for(int i=0;i< x_shp0_usable; i++){
                  int mini_j = 0;
                  int zj = 0;
                  for(int j=0; j<x_shp1_usable; j++){
                    dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)))[0];
                    dtype_%(z)s * __restrict__ z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                    z[0] = (((mini_j|mini_i) == 0) || z[0] < a) ? a : z[0];
                    mini_j = ((mini_j + 1) == %(ds1)s) ? 0 : mini_j+1;
                    zj += (mini_j == 0);
                  }
                  mini_i = ((mini_i + 1) == %(ds0)s) ? 0 : mini_i+1;
                  zi += (mini_i == 0);
                }
              }
            }
        }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 2)


#f = theano.function([d],downsample.max_pool_2d(d,(2,2)))
#print f(e)
p = downsample.DownsampleFactorMax((2,2),True)
e1 = tensor.as_tensor_variable(e1)
t = tensor.as_tensor_variable(t)
h = p.grad([e1],[t])
print h
