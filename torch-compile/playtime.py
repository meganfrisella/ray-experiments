import torch
import torch.nn as nn
import torch.optim as optim

class BaseNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
class PrintNN(BaseNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(PrintNN, self).__init__(input_size, hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        print(x.shape) # print triggers graph break
        x = self.fc2(x)
        return x

class DDControlFlowNN(BaseNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(DDControlFlowNN, self).__init__(input_size, hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if x.mean() > 0:  # Data-dependent control flow
            x = self.fc2(x)
        else:
            x = self.fc3(x)
        return x

class DDControlFlowNN(BaseNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(DDControlFlowNN, self).__init__(input_size, hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if x.mean() > 0:  # Data-dependent control flow
            x = self.fc2(x)
        else:
            x = self.fc3(x)
        return x
    
model = DDControlFlowNN(10, 20, 2)
compiled = torch.compile(model, fullgraph=True)
compiled(torch.randn(1, 10))