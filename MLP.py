import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression, load_data

class HiddenLayer(object):
    
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation = T.tanh):
        
        self.input= input
        
        if W is None:
            W_values =numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),high = numpy.sqrt(6. / (n_in + n_out)),size = (n_in,n_out)),dtype = theano.config.floatX)
            
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
                
            W = theano.shared(value = W_values,name='W',borrow = True)
            
        if b is None:
            b_values = numpy.zeros((n_out,),dtype = theano.config.floatX)
            b = theano.shared(value =b_values,name='b',borrow = True)
            
        self.W = W
        self.b = b
        
        lin_output = T.dot(input,self.W) + self.b
        
        if activation == None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)
        
        self.params = [self.W,self.b]
        
        
class MLP(object):
    
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        
        
        self.hiddenLayer = HiddenLayer(rng = rng,n_in= n_in,input =input,n_out = n_hidden,activation = T.tanh )
        
        self.logRegressionLayer = LogisticRegression(input= self.hiddenLayer.output,n_in = n_hidden,n_out = n_out)
        
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
        
        self.negative_log_likelihood = (self.logRegressionLayer.negative_log_likelihood)
        
        self.errors = self.logRegressionLayer.errors
        
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
        
def test_mlp(learning_rate =0.01,L1_reg = 0.00,L2_reg = 0.0001,n_epochs =1000,dataset = 'mnist.pkl.gz',batch_size = 20,n_hidden = 500):
    
    
    datasets = load_data(dataset)
    
    trainx,trainy = datasets[0]
    validx,validy =datasets[1]
    testx,testy = datasets[2]
    
    #Computing the number of minibatches
    nTrainBatches = trainx.get_value(borrow = True).shape[0] / batch_size
    nValidBatches = validx.get_value(borrow = True).shape[0] / batch_size
    nTestBatches = testx.get_value(borrow = True).shape[0] / batch_size
    
    print("...Building the model...")
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    rng = numpy.random.RandomState(1234)
    
    classifier = MLP(rng = rng, input = x,n_in = 28*28,n_hidden =n_hidden,n_out = 10)
    
    # The cost that the MLP tries to minimize    
    cost = classifier.negative_log_likelihood(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr
    
    test_model = theano.function(inputs=[index],outputs = classifier.errors(y),givens = {x:testx[index*batch_size : (index+1)*batch_size], y:testy[index*batch_size : (index+1)*batch_size]})
    
    valid_model = theano.function(inputs=[index],outputs = classifier.errors(y),givens = {x:validx[index*batch_size : (index+1)*batch_size], y:validy[index*batch_size : (index+1)*batch_size]})
    
    #All the parameter gradients of the MLP are accumulated in gparams. Calculates the gradient of the cost variable with respect to theta
    gparams = [T.grad(cost,param) for param in classifier.params]
    
    #Specifying how to update the parameters of the MLP
    updates = [(param,param- learning_rate*gparam) for param,gparam in zip(classifier.params,gparams)]
    
    
    #Function to train the model
    train_model = theano.function(inputs =[index],outputs = cost,updates = updates,givens ={x : trainx[index*batch_size : (index+1)*batch_size], y : trainy[index*batch_size : (index+1)*batch_size]} )
    
    
    print("..Training the model")
    
    patience = 10000
    patience_increase = 2
    
    improvement_threshold = 0.995
    
    validation_frequency = min(nTrainBatches,patience/2)
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    
    while(epoch<n_epochs) and (not done_looping):
        
        epoch = epoch + 1
        
        for minibatch_index in xrange(nTrainBatches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            
            #iteration number
            iter = (epoch-1)*nTrainBatches + minibatch_index
            
            if(iter+1)%validation_frequency == 0:
                validation_losses = [valid_model(i) for i in xrange(nValidBatches)]
                
                this_validation_loss = numpy.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        nTrainBatches,
                        this_validation_loss * 100.
                    )
                )
                
                
                if this_validation_loss < best_validation_loss:
                    
                    if(this_validation_loss < best_validation_loss*improvement_threshold):
                        
                        patience = max(patience,iter*patience_increase)
                    best_validation_loss=  this_validation_loss
                    best_iter = iter
                    
                    # Testing this model on the test set
                    test_loss = [test_model(i) for i in xrange(nTestBatches)]
                    test_score = numpy.mean(test_loss)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, nTrainBatches,
                           test_score * 100.))
            
            if patience <= iter:
                done_looping = True
                break
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
if __name__ == '__main__':
    test_mlp()
                
                
                
        
    
        
            
        
    
                
        
            