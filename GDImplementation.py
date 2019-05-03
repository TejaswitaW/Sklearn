#gradient decent implementation in pyhton
import numpy as np

def gradient_descent(x,y):
    m_curr=b_curr=0 #we need to start with some initial values
    #then you take some baby steps to reach at global minima
    iterations=1000
    n=len(x)
    #learning rate is a parameter you have to start with some value
    learning_rate=0.001
    #this is the number of baby steps
    for i in range(iterations):
        #we do just trail and error
        y_pred= m_curr * x + b_curr
        cost=(1/n)*sum([ val**2 for val in (y-y_pred)])
        #next step is to calculate m derivative and b derivative
        #n is length of data points.here i am assuming length of x and y is same,
        #if not you have to throw an error
        md= -(2/n)*sum(x*(y-y_pred))
        bd= -(2/n)*sum(y-y_pred)
        m_curr= m_curr - learning_rate * md
        b_curr= b_curr - learning_rate * bd
        print("m: %2.2f, b :%2.2f ,cost: %2.2f ,iteration %d " %(m_curr,b_curr,cost,i))
        #if you want to know how you are perfroming you need to find out cost
        
#where should i stop


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])


gradient_descent(x,y) 
