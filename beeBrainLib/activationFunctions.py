'''
beeBrain - An Artificial Intelligence & Machine Learning library
by Dev. Ibrahim Said Elsharawy (www.devhima.tk)
'''

''''
MIT License

Copyright (c) 2018 Ibrahim Said Elsharawy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from numpy import exp
from constants import Constants
import numpy
from numpy.lib.scimath import logn
from math import e
import enum
from random import uniform

class AFParameters():
	#PRELU
	prelu_alpha = Constants.ALPHA
	
	#RRELU
	rrelu_alpha = Constants.ALPHA
	
	#ELU
	elu_alpha = Constants.ALPHA
	
	#SRELU
	srelu_tl = -0.4
	srelu_al = 0.2
	srelu_tr = 0.4
	srelu_ar = 0.2
	
	#APL
	apl_s = 1
	apl_a = [0.2]
	apl_b = [0.4]
	
	#SoftExp
	softexp_alpha = Constants.ALPHA
	

class ActivationFunctions(enum.Enum):
    
    # >>>> Identity function <<<<
    # The Identity function, is a function that always returns
    # the same value that was used as its argument.
    # In equations, the function is given by f(x) = x.
    @staticmethod
    def identity(x):
        return x
    
    # The derivative of the Identity function, by calculating the first derivative of x.
    # It always returns 1
    @staticmethod
    def identity_derivative(x):
        return 1
    
    #Enum name&value
    IDENTITY = 0

    #____________________________________________________________
    
    # >>>> Binary step (Heaviside step) function <<<<
    # The Binary function, is a function that returns
    # value is 0 for negative argument and 1 for positive argument.
    @staticmethod
    def binary(x):
        return numpy.where(x>=0, 1, 0)
    
    # The derivative of the Binary step (Heaviside step) function, you can get it
    # by calculating the derivative of dH/dx that equals the dirac delta of x.
    # It returns 0 if x>0 or x<0, and it returns ? if x = 0.
    @staticmethod
    def binary_derivative(x):
        return 0
    
    #Enum name&value
    BINARY = 1

    #____________________________________________________________

    # >>>> Sigmoid function <<<<
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    #Enum name&value
    SIGMOID = 2

    #____________________________________________________________

    # >>>> Signum (Softsign) function <<<<
    # The signum function, which extracts the sign of a real number x
    # We pass the weighted sum of the inputs through this function to
    # normalise them between -1 and +1.
    @staticmethod
    def signum(x):
        y = x / (1 + abs(x))
        return y / abs(y)

    # The derivative of the signum function, by calculating y=1 / ((1 + |x|) ^ 2)
    # then calculate y/|y|
    # It returns 1 or 0
    @staticmethod
    def signum_derivative(x):
        y = 1 / ((1 + abs(x)) ** 2)
        return y / abs(y)
        
    #Enum name&value
    SIGNUM = 3

    #____________________________________________________________
    
    # >>>> TanH (Hyperbolic tangent) function <<<<
    # The hyperbolic tangent is the solution to the differential equation 
    # f'=1-f^2 with f(0)=0 and the nonlinear boundary value problem
    @staticmethod
    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    # The derivative of the TanH function, by calculating 1-(f(x))^2
    @staticmethod
    def tanh_derivative(x):
        return 1 - (x ** 2)
        
    #Enum name&value
    TANH = 4

    #____________________________________________________________
    
    # >>>> ArcTan function <<<<
    # We use the ArcTan to inverse trigonometric tangent function
    # and it used to obtain an angle from any of the angle's trigonometric ratios.
    # and we calculate it by tan-1(x)
    @staticmethod
    def arctan(x):
        return numpy.arctan(x)

    # The derivative of the ArcTan function, by calculating 1/((x^2)+1)
    @staticmethod
    def arctan_derivative(x):
        return 1 / ((x ** 2) + 1)
        
    #Enum name&value
    ARCTAN = 5

    #____________________________________________________________

    # >>>> ISRU function <<<<
    # We use the ISRU(Inverse square root unit) to speed up learning in deep neural networks.
    @staticmethod
    def isru(x):
        return x / numpy.sqrt(1 + (Constants.ALPHA * (x ** 2)))

    # The derivative of the ISRU function
    @staticmethod
    def isru_derivative(x):
        return (1 / numpy.sqrt(1 + (Constants.ALPHA * (x ** 2))) ** 3)
        
    #Enum name&value
    ISRU = 6

    #____________________________________________________________

    # >>>> ReLU(Rectified linear unit) function <<<<
    # Used in almost all the convolutional neural networks or deep learning.
    @staticmethod
    def relu(x):
        return numpy.where(x>=0, x, 0)
    
    # The derivative of the ReLU(Rectified linear unit) function.
    @staticmethod
    def relu_derivative(x):
        return numpy.where(x>=0, 1, 0)
    
    #Enum name&value
    RELU = 7

    #____________________________________________________________
    
    # >>>>Leaky ReLU(Leaky rectified linear unit) function <<<<
    # Used in almost all the convolutional neural networks or deep learning.
    # The leak helps to increase the range of the ReLU function.
    # Usually, the value of a is 0.01 or so.
    @staticmethod
    def lrelu(x):
        return numpy.where(x>=0, x, (0.01*x))
    
    # The derivative of the Leaky ReLU(Leaky rectified linear unit) function.
    @staticmethod
    def lrelu_derivative(x):
        return numpy.where(x>=0, 1, 0.01)
    
    #Enum name&value
    LRELU = 8

    #____________________________________________________________
    
    # >>>>PReLU(Parametric rectified linear unit) function <<<<
    # Used in almost all the convolutional neural networks or deep learning.
    # The Parametric helps to increase the range of the ReLU function.
    # Usually, the value of a is ALPHA.
    @staticmethod
    def prelu(x):
        return numpy.where(x>=0, x, (AFParameters.prelu_alpha * x))
    
    # The derivative of the PReLU(Parametric rectified linear unit) function.
    @staticmethod
    def prelu_derivative(x):
        return numpy.where(x>=0, 1, AFParameters.prelu_alpha)
    
    # Parameters setter
    @staticmethod
    def set_prelu_parameters(_alpha):
        AFParameters.prelu_alpha = _alpha
    
    #Enum name&value
    PRELU = 9

    #____________________________________________________________

    # >>>>RReLU(Randomized leaky rectified linear unit) function <<<<
    # Used in almost all the convolutional neural networks or deep learning.
    # The Randomized leak helps to increase the range of the ReLU function.
    # Usually, the value of a is ALPHA.
    @staticmethod
    def rrelu(x):
        return numpy.where(x>=0, x, (AFParameters.rrelu_alpha * x))
    
    # The derivative of the RReLU(Parametric rectified linear unit) function.
    @staticmethod
    def rrelu_derivative(x):
        ActivationFunctions.set_rrelu_parameters(uniform(-Constants.ALPHA, Constants.ALPHA))
        return numpy.where(x>=0, 1, AFParameters.rrelu_alpha)
    
    # Parameters setter
    @staticmethod
    def set_rrelu_parameters(_alpha):
        AFParameters.rrelu_alpha = _alpha
    
    #Enum name&value
    RRELU = 10

    #____________________________________________________________

    # >>>>ELU(Exponential Linear Unit) function <<<<
    # Used in neural networks to speed up learning.
    @staticmethod
    def elu(x):
        return numpy.where(x>=0, x, (AFParameters.elu_alpha * (exp(x) - 1)))
    
    # The derivative of the ELU(Exponential Linear Unit) function.
    @staticmethod
    def elu_derivative(x):
        return numpy.where(x>=0, 1, (x + AFParameters.elu_alpha))
    
    # Parameters setter
    @staticmethod
    def set_elu_parameters(_alpha):
        AFParameters.elu_alpha = _alpha
    
    #Enum name&value
    ELU = 11

    #____________________________________________________________

    # >>>>SELU(Scaled Exponential Linear Unit) function <<<<
    # SELU is some kind of ELU but with a little twist.
    @staticmethod
    def selu(x):
        return numpy.where(x>=0, (Constants.SELU_LAMBDA * x), (Constants.SELU_LAMBDA * (Constants.SELU_ALPHA * (exp(x) - 1))))
    
    # The derivative of the SELU(Scaled Exponential Linear Unit) function.
    @staticmethod
    def selu_derivative(x):
        return numpy.where(x>=0, Constants.SELU_LAMBDA, (Constants.SELU_LAMBDA * (Constants.SELU_ALPHA * exp(x))))
    
    #Enum name&value
    SELU = 12

    #____________________________________________________________

    # >>>>SReLU(S-shaped rectified linear activation unit) function <<<<
    # We use SReLU to learn both convex and non-convex functions,
    # imitating the multiple function forms given by the two fundamental laws,
    # namely the Webner-Fechner law and the Stevens law, in psychophysics and neural sciences.
    @staticmethod
    def srelu(x):
        conditions = [ x<=AFParameters.srelu_tl, (x>AFParameters.srelu_tl)&(x<AFParameters.srelu_tr), x>=AFParameters.srelu_tr ]
        choices = [ AFParameters.srelu_tl + (AFParameters.srelu_al * (x-AFParameters.srelu_tl)), x, AFParameters.srelu_tr + (AFParameters.srelu_ar * (x-AFParameters.srelu_tr)) ]
        result = numpy.select(conditions, choices)
        return result
    
    # The derivative of the SReLU(S-shaped rectified linear activation unit) function.
    @staticmethod
    def srelu_derivative(x):
        conditions = [ x<=AFParameters.srelu_tl, (x>AFParameters.srelu_tl)&(x<AFParameters.srelu_tr), x>=AFParameters.srelu_tr ]
        choices = [ AFParameters.srelu_al, 1, AFParameters.srelu_ar ]
        result = numpy.select(conditions, choices)
        return result
    
    # Parameters setter
    @staticmethod
    def set_srelu_parameters(_tl, _al, _tr, _ar):
        AFParameters.srelu_tl = _tl
        AFParameters.srelu_al = _al
        AFParameters.srelu_tr = _tr
        AFParameters.srelu_ar = _ar
    
    #Enum name&value
    SRELU = 13

    #____________________________________________________________

    # >>>> ISRLU function <<<<
    # We use the ISRLU(Inverse square root linear unit) to speed up learning in deep neural networks.
    @staticmethod
    def isrlu(x):
        return numpy.where(x>=0, x, x / numpy.sqrt(1 + (Constants.ALPHA * (x ** 2))))

    # The derivative of the ISRLU function
    @staticmethod
    def isrlu_derivative(x):
        return numpy.where(x>=0, 1, (1 / numpy.sqrt(1 + (Constants.ALPHA * (x ** 2))) ** 3))
        
    #Enum name&value
    ISRLU = 14

    #____________________________________________________________

    # >>>> APL function <<<<
    # APL(Adaptive piecewise linear) in agriculture
    # piecewise regression analysis of measured data
    # is used to detect the range over which growth factors
    # affect the yield and the range over
    # which the crop is not sensitive to changes in these factors.
    @staticmethod
    def apl(x):
        result = numpy.maximum(0, x)
        for i in xrange(AFParameters.apl_s):
            result += AFParameters.apl_a[i] * numpy.maximum(0, (-x + AFParameters.apl_b[i]))
        return result

    # The derivative of the APL function
    @staticmethod
    def apl_derivative(x):
        result = ActivationFunctions.binary(x)
        result = result.astype('float64')
        for i in xrange(AFParameters.apl_s):
            result -= AFParameters.apl_a[i] * ActivationFunctions.binary(-x + AFParameters.apl_b[i])
        return result
        
    #Enum name&value
    APL = 15

    #____________________________________________________________

    # >>>> SoftPlus function <<<<
    # We use the SP to get Biological plausibility, Sparse activation,
    # Better gradient propagation, Efficient computation, Scale-invariant.
    @staticmethod
    def sp(x):
        return logn(e, 1 + exp(x))

    # The derivative of the SP function
    @staticmethod
    def sp_derivative(x):
        return 1 / (1 + exp(-x))
        
    #Enum name&value
    SP = 16

    #____________________________________________________________

    # >>>> Bent Identity function <<<<
    @staticmethod
    def bi(x):
        return ((numpy.sqrt((x**2)+1) - 1) / 2) + x

    # The derivative of the BI function
    @staticmethod
    def bi_derivative(x):
        return (x / (2 * numpy.sqrt((x**2)+1))) + 1
        
    #Enum name&value
    BI = 17

    #____________________________________________________________

    # >>>> SiLU function <<<<
    @staticmethod
    def silu(x):
        return (x * ActivationFunctions.sigmoid(x))

    # The derivative of the SiLU function
    @staticmethod
    def silu_derivative(x):
        return (x + (ActivationFunctions.sigmoid(x) * (1-x)))
        
    #Enum name&value
    SILU = 18

    #____________________________________________________________
    
    # >>>>SoftExponential(SoftExp) function <<<<
    # This function can exactly calculate many natural operations
    # that typical neural networks can only approximate, 
    # including addition, multiplication, inner product, distance, polynomials, and sinusoids.
    @staticmethod
    def softexp(x):
        conditions = [ AFParameters.softexp_alpha < 0, AFParameters.softexp_alpha == 0, AFParameters.softexp_alpha > 0 ]
        choices = [ -(logn(e, 1 - (AFParameters.softexp_alpha * (x + AFParameters.softexp_alpha))))/AFParameters.softexp_alpha, x, ((exp(AFParameters.softexp_alpha * x)-1)/AFParameters.softexp_alpha) + AFParameters.softexp_alpha ]
        result = numpy.select(conditions, choices)
        return result
    
    # The derivative of the SoftExp function.
    @staticmethod
    def softexp_derivative(x):
        return numpy.where(AFParameters.softexp_alpha>=0, exp(AFParameters.softexp_alpha*x), (1/(1-(AFParameters.softexp_alpha*(AFParameters.softexp_alpha+x)))))
    
    # Parameters setter
    @staticmethod
    def set_softexp_parameters(_alpha):
        AFParameters.softexp_alpha = _alpha
    
    #Enum name&value
    SOFTEXP = 19

    #____________________________________________________________

    # >>>> Sinusoid function <<<<
    # Sinusoid is a mathematical curve that describes a smooth periodic oscillation.
    # A sine wave is a continuous wave. It is named after the function sine, of which it is the graph. 
    # It occurs often in pure and applied mathematics,
    # as well as physics, engineering, signal processing and many other fields.
    @staticmethod
    def sinusoid(x):
        return numpy.sin(x)

    # The derivative of the Sinusoid function.
    @staticmethod
    def sinusoid_derivative(x):
        return numpy.cos(x)
        
    #Enum name&value
    SINUSOID = 20

    #____________________________________________________________

    # >>>> Sinc function <<<<
    @staticmethod
    def sinc(x):
        return numpy.where(x==0, 1, numpy.sin(x)/x)

    # The derivative of the Sinc function
    @staticmethod
    def sinc_derivative(x):
        return numpy.where(x==0, 0, (numpy.cos(x)/x)-(numpy.sin(x)/(x**2)))
        
    #Enum name&value
    SINC = 21

    #____________________________________________________________

    # >>>> Gaussian function <<<<
    @staticmethod
    def gaussian(x):
        return exp(-x ** 2)

    # The derivative of the Gaussian function
    @staticmethod
    def gaussian_derivative(x):
        return -2 * x * exp(-x ** 2)
        
    #Enum name&value
    GAUSSIAN = 22

    #____________________________________________________________
    