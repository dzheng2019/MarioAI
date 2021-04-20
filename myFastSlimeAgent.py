import numpy as np
from numpy import tanh 
import json
from collections import namedtuple
import random

class MyFastModel():

    def __init__(self):
        self.dense1 = np.empty((12,13))
        self.dense2 = np.empty((12,13))
        self.dense3 = np.empty((12,13))
        self.dense4 = np.empty((3,13))
        
        self.l1 = self.dense1.size
        self.l2 = self.l1 + self.dense2.size
        self.l3 = self.l2 + self.dense3.size
        self.l4 = self.l3 + self.dense4.size
        
        self.param_count = self.l4
        
        def find_var(layer):
            dimensions = layer.shape
            shape = (dimensions[0], dimensions[1])
            var = np.random.normal(1, .1, (dimensions[1], dimensions[1]))
            return var;
        
        self.variance1 = find_var(self.dense1)
        self.variance2 = find_var(self.dense2)
        self.variance3 = find_var(self.dense3)
        self.variance4 = find_var(self.dense4)
        
    def predict(self, inputs):
        inputs = np.array([inputs]).T

        # hidden
        x1 = np.append(inputs, [[1]], axis=0)
        z1 = self.dense1 @ self.variance1 @ x1
        a1 = tanh(z1)
        
        # hidden
        x2 = np.append(a1, [[1]], axis=0)
        z2 = self.dense2 @ self.variance2 @ x2
        a2 = tanh(z2)
        
        # hidden
        x3 = np.append(a2, [[1]], axis=0)
        z3 = self.dense3 @ self.variance3 @ x3
        a3 = tanh(z3)
        
        # hidde
        x4 = np.append(a3, [[1]], axis=0)
        z4 = self.dense4 @ self.variance4 @ x4
        a4 = tanh(z4)
        
        return(a4)

    def set_model_params(self,model_params):
        layer1_params = model_params[0:self.l1]
        layer2_params = model_params[self.l1:self.l2]
        layer3_params = model_params[self.l2:self.l3]
        layer4_params = model_params[self.l3:self.l4]
        
        self.dense1 = np.array(layer1_params).reshape((12,13))
        self.dense2 = np.array(layer2_params).reshape((12,13))
        self.dense3 = np.array(layer3_params).reshape((12,13))
        self.dense4 = np.array(layer4_params).reshape((3,13))


if __name__ == '__main__':
    print()
    model = MyFastModel()
    params = [random.random()*2-1 for i in range(model.param_count)]
    print(model.param_count)
    model.set_model_params(params)
    # print(model.dense4)
    print(model.variance1)
    print(model.predict([random.random()*2-1  for _ in range(0,12)]))