# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:50:57 2015

@author: roshan
"""

import theano

import numpy
import theano.tensor as T


from Maxpool import *

d = T.dmatrix('d')
a = numpy.ndarray((4,4))
e=numpy.asarray([[1,2,7,8],[3,4,1,3],[8,5,3,9],[3,4,5,12]])
print e
f = theano.function([d], max_pool_2d(input=d,ds=(2,2),sparsity =1, ignore_border = True) )
print f(e)

