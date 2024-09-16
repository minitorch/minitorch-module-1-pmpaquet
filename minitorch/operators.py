"""Collection of the core mathematical operators used throughout the code base."""

from ast import Call
import math

# ## Task 0.1
from typing import Callable, Iterable

from numpy import iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y:float) -> float:
    """Multiplies two numbers

    Args:
        x: float to be multiplied with y
        y: float to be multiplied with x

    Returns:
        x multiplied by y

    """
    return x * y


def id(x: float) -> float:
    """Identity function: returns the input unchanged

    Args:
        x: a float

    Returns:
        The input (float) unchanged
        
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two floats

    Args:
        x: float to be summed with y
        y: float to be summed with x

    Returns:
        Sum of x and y
        
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number

    Args:
        x: A float to be negated

    Returns:
        x negated
        
    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if first input is less than the second input

    Args:
        x: float
        y: float

    Returns:
        boolean specifying whether x is less than y

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks whether the two inputs are equal

    Args:
        x: float
        y: float

    Returns:
        Boolean specifying whether x and y are equal

    """
    return x == y


def max(x: float, y: float) -> float:
    """Calculates the maximum of the two inputs

    Args:
        x: float
        y: float

    Returns:
        (float) the maximum of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Returns whether x and y are with 0.01 of each other
    
    Args:
        x: float
        y: float
        
    Returns:
        (bool) whether x and y are close

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Returns the sigmoid function of the input

    Args:
        x: float

    Returns:
        (float) Sigmoid(x)

    """
    ans: float
    if x >= 0.:
        ans = 1. / (1. + math.exp(-x))
    else:
        ans = math.exp(x) / (1. + math.exp(x))
    return ans


def relu(x: float) -> float:
    """Applies the ReLU activation function to the input
    
    Args:
        x: float
        
    Returns:
        ReLU(x) (float)   

    """
    return x if x > 0. else 0.


def log(x: float) -> float:
    """Applies the natural logarithm to the input

    Args:
        x: float

    Returns:
        float: log(x)

    """
    return math.log(x)


def exp(x: float) -> float:
    """Applies the exponential function to the input

    Args:
        x: float

    Returns:
        float: exp(x)

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of the input

    Args:
        x: float
    
    Returns:
        (float) The reciprocal of x

    """
    return 1. / x


def log_back(x: float, y: float) -> float:
    """Returns the derivative of the log function of times another argument
    
    Args:
        x: take d/dx log 
        y: argmuent to multiple to d/dx log

    Returns:
        (float) the reciprocal of x times y

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Returns the derivative of the inverse function of the input times another arg
    
    Args:
        x: float

    Returns:
        (float) The derivative of the inverse function

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Returns the derivative of ReLU of the first input times second argument

    Args:
        x: float
        y: float

    Returns:
        (float) derivative of ReLU on x times y

    """
    return y if x >= 0. else 0.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable) -> Callable:
    """Higher-order function Applies a function to every element in an iteratable
    
    Args:
        func: Callable to apply over the iterable
        
    Returns:
        Iterable containing results of applying funtion to iterable
    
    """
    return lambda it: [func(elm) for elm in it]


def zipWith(func: Callable) -> Callable:
    """Higher-order function that combines two iterables with a given function
    
    Args:
        func: Callable to apply to elements of both iterables
        
    Returns:
        Callable to compute [func(a,b) for a,b in zip(it_a, it_b)]
        
    """
    return lambda iterA, iterB : [func(a, b) for a,b in zip(iterA, iterB)]


def reduce(func: Callable) -> Callable:
    """Higher-order function that uses a given funtion to reduce an iterable
    to a single element.
    
    Args:
        func (Callable): function that reduces all elements in the iterable
    
    Returns:
        Callable that applies the funciton to reduce an iterable to a single
        element
        
    """
    def _reducer(iter: Iterable) -> float:
        ans: float = 0.
        for i,val in enumerate(iter):
            ans = func(ans, val) if i else val
        return ans
    return _reducer


def negList(data: Iterable) -> Iterable:
    """Negate all elements in a list
    
    Args:
        data (Iterable): list of elements to negate
        
    Returns:
        (Iterable) list of negated elements from data
        
    """
    return map(neg)(data)


def addLists(iterA: Iterable, iterB: Iterable) -> Iterable:
    """Creates a list of element-wise additions from input iterables
    
    Args:
        iterA (Iterable): List of elements to add with iterB
        iterB (Iterable): List of elements to add with iterA
        
    Returns:
        (Iterable) list of element-wise sums of iterA and iterB
        
    """
    return zipWith(add)(iterA, iterB)


def sum(iter: Iterable) -> float:
    """Sums all elements in input iterable
    
    Args:
        iter (Iterable): data to sum
        
    Returns:
        (float) sum of all elements in input iterable
        
    """
    return reduce(add)(iter)


def prod(iter: Iterable) -> float:
    """Computes product of all elements in input iterable
    
    Args:
        iter (Iterable): data to multiply
        
    Returns:
        (float) product of all elements in input iterable
        
    """
    return reduce(mul)(iter)
