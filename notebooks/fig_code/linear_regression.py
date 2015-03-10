import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_linear_regression():
    
    #generate some points that follow a theoretical line
    a = 0.5
    b = 1.0

    # 20 random numbers ranging from [0,30]
    x = 30 * np.random.random(20)

    # y = a*x + b with noise
    y = a * x + b + np.random.normal(size=x.shape)
    

    # create a linear regression classifier
    clf = LinearRegression()
    #compute coefficients a,b for the best fit line y=ax+b
    clf.fit(x[:, None], y)

    # predict y from the data
    #create an array of 100 element ranging from [0,30]
    x_new = np.linspace(0, 30, 100)
    #predict y coordinate of x_new points using our computed model
    y_new = clf.predict(x_new[:, None])

    # plot the results
    ax = plt.axes()
    #plot the data points
    ax.scatter(x, y)
    #plot the regression line
    ax.plot(x_new, y_new)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.axis('tight')


if __name__ == '__main__':
    plot_linear_regression()
    plt.show()
