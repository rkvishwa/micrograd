from graph import draw_dot
import math
import random

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.gradient = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
            
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** (-1))
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int or float for the power for now"
        out = Value(self.data ** other, (self, ), f"**{other}")
        
        def _backward():
            self.gradient += other * (self.data ** (other - 1)) * out.gradient
            
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.gradient += out.data * out.gradient
            
        out._backward = _backward
        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.gradient += (1 - t**2) * out.gradient
            
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.gradient = 1.0
        
        for node in reversed(topo):
            node._backward()
            
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-0.1, 0.1), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")
        self.nonlin = nonlin
        
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x) if xi != 0.0), self.b)
        return act.tanh() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
      
      
if __name__ == "__main__":
    n = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    def train(steps = 20, step_size = 0.05):
        for k in range(steps):
            ypred = [n(x) for x in xs]
            loss = sum((yout - yreq) ** 2 for yreq, yout in zip(ys, ypred))
            
            for p in n.parameters():
                p.gradient = 0.0
            loss.backward()
            
            for p in n.parameters():
                p.data += -step_size * p.gradient
                
            print(k, loss.data)
            

    train(250, 0.05)
    train(250, 0.01)
    train(10000, 0.001)

    test_xs = [
        [2.5, 2.8, -0.9],
        [0.4, 1.2, 0.9],
    ]

    print("\n-----Testing-----")

    for x in test_xs:
        prediction = n(x)
        
        predicted_class = "+1.0" if prediction.data > 0 else "-1.0"
        print(f"Input: {x} -> Raw Output: {prediction.data:.4f} | Class: {predicted_class}")