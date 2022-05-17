import psutil, random
import matplotlib.pyplot as plt
import pandas as pd

def dataset():
    random.seed(int(psutil.cpu_percent(1)*psutil.virtual_memory()[3]%psutil.virtual_memory()[4]))
    
    points = [[],[]]

    for i in range(1,10):
        points[0].append(i)
        points[1].append(random.randint(0,10))

    print(points)

    plt.plot(points[0], points[1])
    plt.show()
    
    return points

    

if __name__ == "__main__":
	dataset()