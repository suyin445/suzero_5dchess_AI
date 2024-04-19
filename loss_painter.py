import json
import numpy as np
import matplotlib.pyplot as plt

with open(r'./save/suzero_loss.json', 'a+') as f:
    #loss = json.load(f)
    Y = np.array([0.13670535385608673, 0.13670535385608673, 0.77452152967453, 0.4722501039505005, 0.2016112506389618,
                  0.018303317949175835, 0.04416488856077194, 0.04069380462169647, 0.0586092472076416,
                  0.018531179055571556])
    X = np.array(range(len(Y)))
    plt.plot(X, Y)
    plt.show()
