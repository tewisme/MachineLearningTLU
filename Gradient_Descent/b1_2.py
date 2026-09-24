def grad1(x):
    return 2*x

def grad2(x):
    return x**2 - 1

def cost1(x):
    return x**2 - 2

def cost2(x):
    return 1.0*(x**3)/3 - x

def grad_descent(x0, learning_rate):
    x = [x0]
    for it in range(50):
        x.append(x[-1] - learning_rate*grad2(x[-1]))
    return (x[-1], it)

(x1, it1) = grad_descent(2, 0.1)
print(x1, it1)
