from lasagne.objectives import squared_error as sq
from lasagne.objectives import aggregate


def squared_error(network_output, target):
    ''' squared error of lasagne.objective'''
    return sq(network_output, target)


def partial_squared_error(network_output, target):
    ''' Return the partial squared error

    In particular, when calculating the loss,
    return only the value of those that are different
    from -1, i.e. the points we want to make prediction
    for !!!

    '''

    cost = (network_output[(target > -1).nonzero()] -
            target[(target > -1).nonzero()])**2
    return cost
