```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
%matplotlib inline
```


```python
transform = transforms.ToTensor()
```

### MNIST Dataset 


```python
train_data = datasets.MNIST(root='./Data/', train=True, download=True, transform=transform)
```


```python
test_data = datasets.MNIST(root='./Data/', train=False, download=True, transform=transform)
```


```python
image, label = train_data[0]
```


```python
plt.imshow(image.reshape(28,28), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1d315dfe208>




    
![png](BasicNN_files/BasicNN_6_1.png)
    



```python
torch.manual_seed(101)
```




    <torch._C.Generator at 0x1d30ceae330>



### Setting a batch size for training and test data


```python
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
```


```python
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)
```


```python
from torchvision.utils import make_grid
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}' ))
```


```python
for images, labels in train_loader: # Loading a batch
    break
```


```python
labels.shape
```




    torch.Size([100])




```python
print('Labels: ', labels[:12].numpy())
im = make_grid(images[:12], nrow=12)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))
```

    Labels:  [   0    5    7    8    6    7    9    7    1    3    8    4]
    




    <matplotlib.image.AxesImage at 0x1d315eaaa58>




    
![png](BasicNN_files/BasicNN_14_2.png)
    


### MultilayerPerceptron
- 28x28x1 input neurons, 10 output neurons
- Default Inners 120, 84
- FC layers ReLu Activations
- Output Log softmax


```python
class MultilayerPerceptron(nn.Module):
    
    # Fully connected layers
    # Relu Feed Forward
    # Multi-class classification activation function (softmax)

    def __init__(self, in_sz=784,out_sz=10,layers=[120,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)
```

### Found good performance with 392 -> 196 inner layers


```python
torch.manual_seed(101)
model = MultilayerPerceptron(784, 10, [392, 196])
```

### Number of total connections between each layer


```python
for i, param in enumerate(model.parameters()):
    if i == 0:
        print(model.fc1.in_features)
    print(param.numel())
```

    784
    307328
    392
    76832
    196
    1960
    10
    


```python
criterion = nn.CrossEntropyLoss() # Loss minimization function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Optimizer steps
```


```python
images.view(100, -1).shape
```




    torch.Size([100, 784])



### Training


```python
start_time = time.time()

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%200 == 0:
            accuracy = trn_corr.item()*100/(100*b)
            print(f'Epoch {i} batch {b} loss {loss.item} accuracy: {accuracy}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Test unseen
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500, -1))
            predicted = torch.max(y_val, 1)[1]
            tst_corr += (predicted==y_test).sum()
    
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time/60} mins')
```

    Epoch 0 batch 200 loss <built-in method item of Tensor object at 0x000001D3257A8048> accuracy: 99.8
    Epoch 0 batch 400 loss <built-in method item of Tensor object at 0x000001D3160E0AB0> accuracy: 99.6725
    Epoch 0 batch 600 loss <built-in method item of Tensor object at 0x000001D316633C18> accuracy: 99.73166666666667
    Epoch 1 batch 200 loss <built-in method item of Tensor object at 0x000001D3160E0AB0> accuracy: 99.825
    Epoch 1 batch 400 loss <built-in method item of Tensor object at 0x000001D31609E1B0> accuracy: 99.865
    Epoch 1 batch 600 loss <built-in method item of Tensor object at 0x000001D31656D948> accuracy: 99.815
    Epoch 2 batch 200 loss <built-in method item of Tensor object at 0x000001D3160CD828> accuracy: 99.635
    Epoch 2 batch 400 loss <built-in method item of Tensor object at 0x000001D3160A5828> accuracy: 99.6925
    Epoch 2 batch 600 loss <built-in method item of Tensor object at 0x000001D31605F900> accuracy: 99.71166666666667
    Epoch 3 batch 200 loss <built-in method item of Tensor object at 0x000001D3160A5828> accuracy: 99.925
    Epoch 3 batch 400 loss <built-in method item of Tensor object at 0x000001D316532C18> accuracy: 99.925
    Epoch 3 batch 600 loss <built-in method item of Tensor object at 0x000001D3254BDBD0> accuracy: 99.90833333333333
    Epoch 4 batch 200 loss <built-in method item of Tensor object at 0x000001D31653BB40> accuracy: 99.835
    Epoch 4 batch 400 loss <built-in method item of Tensor object at 0x000001D31653BC18> accuracy: 99.815
    Epoch 4 batch 600 loss <built-in method item of Tensor object at 0x000001D3160CD828> accuracy: 99.78833333333333
    Epoch 5 batch 200 loss <built-in method item of Tensor object at 0x000001D31653BB40> accuracy: 99.705
    Epoch 5 batch 400 loss <built-in method item of Tensor object at 0x000001D3160E0AB0> accuracy: 99.72
    Epoch 5 batch 600 loss <built-in method item of Tensor object at 0x000001D31653BC18> accuracy: 99.74833333333333
    Epoch 6 batch 200 loss <built-in method item of Tensor object at 0x000001D3167B9E10> accuracy: 99.925
    Epoch 6 batch 400 loss <built-in method item of Tensor object at 0x000001D3167B9E10> accuracy: 99.9025
    Epoch 6 batch 600 loss <built-in method item of Tensor object at 0x000001D3167B9E10> accuracy: 99.89166666666667
    Epoch 7 batch 200 loss <built-in method item of Tensor object at 0x000001D3167B97E0> accuracy: 99.8
    Epoch 7 batch 400 loss <built-in method item of Tensor object at 0x000001D3167B97E0> accuracy: 99.81
    Epoch 7 batch 600 loss <built-in method item of Tensor object at 0x000001D3167B97E0> accuracy: 99.775
    Epoch 8 batch 200 loss <built-in method item of Tensor object at 0x000001D3167B9AB0> accuracy: 99.8
    Epoch 8 batch 400 loss <built-in method item of Tensor object at 0x000001D3167B9AB0> accuracy: 99.8075
    Epoch 8 batch 600 loss <built-in method item of Tensor object at 0x000001D3167B9AB0> accuracy: 99.80833333333334
    Epoch 9 batch 200 loss <built-in method item of Tensor object at 0x000001D3167B9BD0> accuracy: 99.9
    Epoch 9 batch 400 loss <built-in method item of Tensor object at 0x000001D3167B9BD0> accuracy: 99.875
    Epoch 9 batch 600 loss <built-in method item of Tensor object at 0x000001D3167B9BD0> accuracy: 99.85
    Duration: 1.4264329393704733 mins
    

### Loss and test accuracy over epochs


```python
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1d3257c40b8>




    
![png](BasicNN_files/BasicNN_26_1.png)
    



```python
train_acc = [t/600 for t in train_correct]
test_acc = [t/100 for t in test_correct]
```


```python
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1d316573d30>




    
![png](BasicNN_files/BasicNN_28_1.png)
    


### Unseen test data with no gradient change


```python
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
```


```python
with torch.no_grad():
    correct = 0

    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
```


```python
100* correct.item()/len(test_data)
```




    97.96




```python
confusion_matrix(predicted.view(-1), y_test.view(-1))
```




    array([[ 970,    1,    4,    0,    2,    1,    2,    1,    3,    0],
           [   1, 1127,    2,    0,    0,    0,    3,    2,    0,    2],
           [   1,    0, 1006,    2,    1,    0,    0,    5,    1,    0],
           [   1,    1,    6,  995,    2,   18,    1,    0,    7,    5],
           [   0,    0,    3,    0,  957,    2,    8,    0,    2,    5],
           [   0,    1,    0,    4,    0,  849,    3,    0,    4,    0],
           [   2,    1,    1,    0,    2,    4,  939,    0,    0,    0],
           [   1,    2,    6,    4,    0,    2,    0, 1010,    4,    1],
           [   1,    2,    4,    2,    0,    9,    2,    2,  948,    1],
           [   3,    0,    0,    3,   18,    7,    0,    8,    5,  995]],
          dtype=int64)


