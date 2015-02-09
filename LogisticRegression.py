import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    
    def __init__(self,input,n_in,n_out):
        
        
        self.W = theano.shared(value = numpy.zeros((n_in,n_out),dtype = theano.config.floatX),borrow =True,name ='W')
        
        self.b = theano.shared(value = numpy.zeros((n_out,),dtype = theano.config.floatX),name = 'b',borrow = True)
        
        self.p_y_givenx = T.nnet.softmax(T.dot(input,self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_givenx,axis=1)
        
        self.params = [self.W,self.b]
        
    def negative_log_likelihood(self,y):
            
            
        return -T.mean(T.log(self.p_y_givenx)[T.arange(y.shape[0]),y])
            
            
    def errors(self,y):
            
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
                
    
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    print(data_dir)
    print(data_file)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        print("new path is "+new_path)
        new_path = "/Documents/mnist.pkl.gz"
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

    
def sgd_optimization(learning_rate = 0.13,n_epochs = 1000,dataset = 'mnist.pkl.gz',batch_size = 600):
    
    
    datasets = load_data(dataset)
    
    trainx,trainy = datasets[0]
    validx,validy = datasets[1]
    testx,testy = datasets[2]
    
    n_trainBatches = trainx.get_value(borrow = True).shape[0]/batch_size
    n_validBatches = validx.get_value(borrow= True).shape[0]/batch_size
    n_testBatches = testx.get_value(borrow= True).shape[0]/batch_size
    
    print("...Building model...")
    
    index = T.lscalar()   #indicates the index of the minibatch
    
    x = T.matrix('x')
    y = T.ivector('y')
    
    
    classifier = LogisticRegression(input = x,n_in = 28*28,n_out = 10)
    
    cost = classifier.negative_log_likelihood(y)
    
    test_model = theano.function(inputs = [index],outputs = classifier.errors(y),givens = {x: testx[index*batch_size : (index+1)*batch_size],y: testy[index*batch_size : (index+1)*batch_size]})
    
    validate_model = theano.function(inputs = [index],outputs = classifier.errors(y),givens = {x: validx[index*batch_size : (index+1)*batch_size],y: validy[index*batch_size : (index+1)*batch_size]})
    
    # Computing the gradients of ost witht respect to Theta = (W,b)
    gradW = T.grad(cost = cost,wrt = classifier.W)
    gradb = T.grad(cost = cost,wrt = classifier.b)
    
    #The changes that are to be made are made through the updates. It specifies how to update the parameters
    updates = [(classifier.W,classifier.W - learning_rate*gradW),(classifier.b,classifier.b - learning_rate*gradb)]
    
    train_model = theano.function(inputs = [index],outputs = cost,updates = updates,givens = {x: trainx[index*batch_size : (index+1)*batch_size],y: trainy[index*batch_size : (index+1)*batch_size]})
    
    
    ## TRAINING THE MODEL
    print("Training the model")
    
    patience = 5000 # Look at these many examples minimum
    patience_increase = 2
    
    improvement_threshold = 0.995
    
    validation_frequency = min(n_trainBatches,patience/2)
    
    best_validation_loss = numpy.inf
    
    test_score = 0
    start_time = time.clock()
    
    
    done_looping = False
    epoch = 0
    
    while(epoch<n_epochs) and (not done_looping):
        
        epoch = epoch+1
        for minibatch_index in xrange(n_trainBatches):
            minibatch_avgCost =train_model(minibatch_index)
            
            iter = (epoch-1)*n_trainBatches + minibatch_index
            
            if (iter+1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_validBatches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_trainBatches,
                        this_validation_loss * 100.
                    )
                
                )
            
                if this_validation_loss<best_validation_loss:
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience,iter*patience_increase)
                    
                    best_validation_loss= this_validation_loss
                
                    test_losses = [test_model(i) for i in xrange(n_testBatches)]
                    test_score = numpy.mean(test_losses)
                
                    print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_trainBatches,
                                test_score * 100.
                            )
                        )
                        
            if patience <= iter:
                done_looping = True
                break
            
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
                          
    
if __name__ == '__main__':
    
    sgd_optimization()

        
                    
            

                
              
    
    
                