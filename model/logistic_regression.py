import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x,feats=None):
        keep_feats = isinstance(feats,list)
        x = x.view(-1, 28*28)
        if keep_feats:
            feats.append(x)
        x = torch.relu(self.fc1(x))
        if keep_feats:
            feats.append(x)
        x = torch.relu(self.fc2(x))
        if keep_feats:
            feats.append(x)
        x = self.fc3(x)
        if keep_feats:
            
            feats.append(x)
        return x
    def forward_with_jitter(self, x,weight_jitter):
        assert len(weight_jitter) == TODO
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x) + TODO.torch.nn.functional.linear(x,weight_jitter[0]))
        x = torch.relu(self.fc2(x) + TODO.linear(x,weight_jitter[1]))
        x = self.fc3(x)+ TODO.linear(x,weight_jitter[2])
        return x    
    # """
    def forward_and_backward(self,x,y,get_dloss_by_dout):
        feats = []
        out = self.forward(x,feats=feats)
        # assert dloss_by_dout.shape == out.shape
        # loss = get_loss(out,y)
        print('TODO:put in the code for the linear layer')
        g_wrt_output = get_dloss_by_dout(y)
        # modules: [Linear,ReLU,Linear,ReLU,Linear,ReLU]
        grads = {}
        for li,l in enumerate(reversed(self.modules())):
            if isinstance(l,torch.ReLU):
                # raise NotImplementedError
                mask = (feats[li] > 0).float()
                g_wrt_output = g_wrt_output * mask
            if isinstance(l,torch.Linear):
                # input: 10,100
                # grad_wrt_output: 10,256
                # weights: 256x100
                # bias: 256
                assert li >= 1
                input = feats[li-1]
                g_wrt_bias = torch.ones_like(l.bias)
                #,(10,256),(256,100) ->(10,100)
                assert l.weights.shape[0] == feats[li].shape[-1]
                g_wrt_input = torch.einsum('ij,jk->ik',g_wrt_output,l.weights)
                10,256
                # raise NotImplementedError
                #(10,256),(10,100)->(10,256,100)
                g_wrt_w = torch.einsum('ij,ik->ijk',g_wrt_output,input)
                g_wrt_output = g_wrt_input
                grads[li] = {'weights':g_wrt_w,'bias':g_wrt_bias}
        return out,grads
    # """
    """
    def forward_and_backward(self,x,y,criterion):
        feats = []
        out = self.forward(x,feats=feats)
        loss = criterion(out,y)
        for p in self.parameters():
            if p.grad:
                p.grad.zero_()
        loss.backward()
        grads = [p.grad for p in self.parameters()]
        
        return out,grads,loss
    """
# Define a loss function and optimizer
model = MLP()
model_no_grad = MLP()
def synchronize_models(ref=None,clone=None):
    for rp,cp in zip(ref.parameters(),clone.parameters()):
        cp.data.copy_(rp)
synchronize_models(ref=model,clone=model_no_grad)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Define a DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

# Train the model
def train(epoch,model,model_no_grad):
    model.train()
    synchronize_models(ref=model,clone=model_no_grad)
    for data, target in train_loader:
        model.train()
        model_no_grad.eval()
        
        optimizer.zero_grad()
        print('TODO: define lr as a parameter')
        lr = TODO.1e-3
        output,parameter_grads,loss1 = model.forward_and_backward(TODO.fake_data,TODO.fake_target,criterion)
        TODO:
            TODO.model1.forward_and_backward(data,target,get_dloss_by_dout)
        # TODO.model2.eval()
        print('TODO: parameter grads is a list')
        jittered_parameters = [lr * p  for p in parameter_grads]
        output,parameter_grads,loss = TODO.model_no_grad.forward_with_jitter(data,jittered_parameters)
        print('TODO:add the fake data generator')
        print('TODO:add criterion derivative wrt output')
        # output = model(data)
        loss = criterion(output, target)
        loss.backward()
        print('TODO:this should only be the optimizer for model not model_no_grad')
        optimizer.step()

# Test the model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
    train(epoch)
    test()

