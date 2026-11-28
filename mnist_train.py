import random
import json
from sklearn.datasets import fetch_openml
from lib import Value, Neuron, Layer, MLP

print("---Loading MNIST Dataset---")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

X_all = (mnist.data / 255.0).tolist()
Y_all = [int(y) for y in mnist.target]

X_train, Y_train = X_all[:5000], Y_all[:5000]
X_test, Y_test = X_all[5000:6000], Y_all[5000:6000]

def one_hot(digit):
    vec = [0.0] * 10
    vec[digit] = 1.0
    return vec

Y_train_oh = [one_hot(y) for y in Y_train]

model = MLP(784, [16, 16, 10])
print(f"Total parameters: {len(model.parameters())}")

batch_size = 25
epochs = 12
start_lr = 0.05
end_lr = 0.01

for epoch in range(epochs):
    progress = epoch / max(1, epochs - 1)
    learning_rate = start_lr + (end_lr - start_lr) * progress
    
    combined = list(zip(X_train, Y_train_oh))
    random.shuffle(combined)
    X_shuf, Y_shuf = zip(*combined)
    
    total_loss = 0.0
    num_batches = len(X_train) // batch_size
    
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        xb = X_shuf[start:end]
        yb = Y_shuf[start:end]
        
        ypred = [model(x) for x in xb]
        
        loss = sum(
            (yout - ygt) ** 2
            for y_row, gt_row in zip(ypred, yb)
            for yout, ygt in zip(y_row, gt_row)
        ) / batch_size
        
        for p in model.parameters():
            p.gradient = 0.0
        
        loss.backward()
        
        for p in model.parameters():
            p.data += -learning_rate * p.gradient
            
        total_loss += loss.data
        
    
    correct = 0
    for x, true_digit in zip(X_test, Y_test):
        scores = [out.data for out in model(x)]
        if scores.index(max(scores)) == true_digit:
            correct += 1
            
    accuracy = (correct / len(X_test)) * 100
    print(f"Epoch {epoch + 1}/{epochs} | Avg loss: {total_loss/num_batches:.4f} | Test accuracy: {accuracy:.1f}%")
        
        
with open("mnist_weights.json", "w") as f:
    json.dump([p.data for p in model.parameters()], f)
    
print("Saved trained weights to mnist_weights.json")