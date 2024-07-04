import torch
net = torch.nn.Sequential(
    torch.nn.Linear(16,32),
    torch.nn.GELU(),
    torch.nn.Linear(32,1)
)
torch.random.manual_seed(42)
x = torch.randn(4,16)
y = torch.randn(4,1)
net.zero_grad()
yhat = net(x)
loss = torch.nn.functional.mse_loss(yhat,y)
loss.backward()
print(net[0].weight.grad.view(-1)[:10])

net.zero_grad()
for i in range(x.shape[0]):
    yhat = net(x[i])
    loss = torch.nn.functional.mse_loss(yhat,y[i])
    loss = loss/x.shape[0]
    loss.backward()
print(net[0].weight.grad.view(-1)[:10])


