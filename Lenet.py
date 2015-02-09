import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from LogisticRegression import LogisticRegression, load_data
from MLP import HiddenLayer

class LeNetConvPoolLayer(object):
    
    def __init__(self,rng,input,filter_shape,image_shape,poolsize =(2,2)):
        
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        fan_in = numpy.prod(filter_shape[1:])
        
        fan_out = (filter_shape[0]*numpy.prod(filter_shape[2:])/numpy.prod(poolsize))
        
        W_bound = numpy.sqrt(6./(fan_in + fan_out))
        
        
        #Initializing the random weights
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound,high = W_bound,size = filter_shape),dtype = theano.config.floatX),borrow = True)
        
        #Initializing the random weight
        
        b_values = numpy.zeros((filter_shape[0],),dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values,borrow =True)
        
        #Convolving input feature map with filters
        conv_out = conv.conv2d(input = input,filters = self.W, filter_shape = filter_shape,image_shape = image_shape)
        
        #Downsampling each of the feature maps.Here we take the max pixel value of a sampling as the output
        pooled_out = downsample.max_pool_2d(input = conv_out,ds = poolsize,ignore_border = True)
        
        
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        
        self.params = [self.W,self.b]
        
        
        
def runNN(learning_rate= 0.1, n_epochs = 200,dataset='mnist.pkl.gz',nkerns = [20,50],batch_size = 500):
    
    
    rng =   numpy.random.RandomState(1234)
    
    datasets = load_data(dataset)
    
    trainx,trainy = datasets[0]
    validx,validy = datasets[1]
    testx,testy = datasets[2]
    
    nTrainBatches = trainx.get_value(borrow=True).shape[0]/batch_size
    nValidBatches = validx.get_value(borrow=True).shape[0]/batch_size
    nTestBatches = testx.get_value(borrow=True).shape[0]/batch_size
    
    index = T.iscalar()
    
    x = T.matrix('x')
    y = T.ivector('y')
    
    ##Building the model
    print("...BUILDING THE MODEL...")
    
    layer0_input = x.reshape((batch_size,1,28,28))
    
    layer0 = LeNetConvPoolLayer(rng,input = layer0_input,image_shape = (batch_size,1,28,28),filter_shape=(nkerns[0],1,5,5),poolsize = (2,2))
    
    #2nd Convolutional layer    
    layer1 = LeNetConvPoolLayer(rng,input = layer0.output,image_shape = (batch_size,nkerns[0],12,12),filter_shape = (nkerns[1],nkerns[0],5,5),poolsize = (2,2))

    layer2_input = layer1.output.flatten(2)    
    
    #Fully connected layer
    layer2 = HiddenLayer(rng,input = layer2_input,n_in = nkerns[1]*4*4,n_out = 500,activation = T.tanh)
    
    #Softmax layer
    layer3 = LogisticRegression(input = layer2.output,n_in = 500,n_out = 10)
    
    #cost function
    cost = layer3.negative_log_likelihood(y)
    
    test_model = theano.function([index],layer3.errors(y),givens = {x:testx[index*batch_size : (index+1)*batch_size], y:testy[index*batch_size : (index+1)*batch_size]})
    
    validate_model = theano.function([index],layer3.errors(y),givens = {x:validx[index*batch_size : (index+1)*batch_size] , y:validy[index*batch_size : (index+1)*batch_size]})
    
    #Parameters of the entire model
    params = layer0.params + layer1.params + layer2.params + layer3.params
    

    grads = T.grad(cost,params)
    
    
    #the updates to be made to the weights
    updates = [(param,param- learning_rate*grad) for param,grad in zip(params,grads)]
    
    #trainig function    
    train_model = theano.function([index],cost,updates = updates,givens = {x:trainx[index*batch_size : (index+1)*batch_size], y:trainy[index*batch_size : (index+1)*batch_size]})
    
    
    print("TRAINING THE MODEL...")
    
    patience = 10000
    patience_increase = 2
    improvement_threshold= 0.995
    
    validation_frequency = min(nTrainBatches,patience/2)
    
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = time.clock()
    
    
    epoch = 0
    done_looping = False
    
    while(epoch<n_epochs) and (not done_looping):
        epoch  = epoch + 1
        for minibatch_index in xrange(nTrainBatches):
            iter = (epoch-1)*nTrainBatches + minibatch_index
            
            if (iter+1)%100 == 0:
                #computing the loss
                print("training at iter = ",iter)
                
                cost = train_model(minibatch_index)
            if (iter+1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(nValidBatches)]
                this_validation_loss  = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %(epoch, minibatch_index + 1, nTrainBatches, this_validation_loss * 100.))
                
                if this_validation_loss < best_validation_loss:
                    
                    if this_validation_loss < best_validation_loss * improvement_threshold :
                        patience = max(patience,iter*patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    #testing the model on test data
                    test_losses = [test_model(i) for i in xrange(nTestBatches)]
                    
                    test_score  = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %(epoch, minibatch_index + 1, nTrainBatches, test_score * 100.))
                    
            if patience <= iter:
                done_looping = True
                break
            
    end_time =time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    runNN()
                    
                    

                