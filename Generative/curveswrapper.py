"""
MIT License
Copyright (c) 2020 Octavio Gonzalez-Lugo 
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
@author: Octavio Gonzalez-Lugo
"""

###############################################################################
# Wrapper Functions
###############################################################################

def Wrapper01(x,y):
    return x**2+np.pi*np.sin(y)

def Wrapper02(x,y):
    return y+np.pi*np.cos(x**2)
    
def Wrapper01(x,y):
    return x+np.abs(np.pi*np.sin(y))

def Wrapper02(x,y):
    return y+np.abs(np.pi*np.cos(x))
    
def Wrapper01(x,y):
    return x**2+y**2

def Wrapper02(x,y):
    return np.cos(x)+np.sin(y)

def Wrapper01(x,y):
    return x**2+y**2

def Wrapper02(x,y):
    return np.sin((x-y))+y

def Wrapper01(x,y):
    return np.sin(x)+np.cos(y)+x

def Wrapper02(x,y):
    return np.sin((x-y))+y


def Wrapper01(x,y):
    return np.cos(x)-y

def Wrapper02(x,y):
    return np.sin(y)-x

def Wrapper01(x,y):
    return x+np.cos(x-y)*y

def Wrapper02(x,y):
    return y+np.sin(y-x)*x
    

def Wrapper01(x,y):
    return x-np.cos(x/y)

def Wrapper02(x,y):
    return y+np.sin(y*x)

def Wrapper01(x,y):
    return np.sin(x-y)+x

def Wrapper02(x,y):
    return np.sin(y+x)+y

def Wrapper01(x,y):
    return np.sin(x-y)+((-1)**np.sign(np.sin(x-y)))*x

def Wrapper02(x,y):
    return np.sin(y+x)+y

def Wrapper01(x,y):
    return np.sin(x)*np.cos(y)

def Wrapper02(x,y):
    return np.sin(y-x)

def Wrapper01(x,y):
    return np.sin(x)+np.cos(y)

def Wrapper02(x,y):
    return np.sin(y)*np.cos(x)
    
def Wrapper01(x,y):
    return x+x*np.sin(x+y)

def Wrapper02(x,y):
    return np.sin(y-x)*x
    
def Wrapper01(x,y):
    return x+x*np.sin(x+y)

def Wrapper02(x,y):
    return np.sin(y)*x
    
def Wrapper01(x,y):
    return x+x*np.sin(x+y)

def Wrapper02(x,y):
    return np.cos(y)+y*x

def Wrapper01(x,y):
    return x+x*np.sin(x+y)

def Wrapper02(x,y):
    return np.cos(y)+x

def Wrapper01(x,y):
    return np.sin(y)*np.sin(x)

def Wrapper02(x,y):
    return x*np.cos(y)

def Wrapper01(x,y):
    return x+np.sin(y)*np.sin(x)

def Wrapper02(x,y):
    return y+np.cos(x)*np.cos(y)
