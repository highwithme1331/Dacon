#LeNet-5
x = torch.randn(1, 1, 32, 32)

conv1 = nn.Conv2d(1, 6, kernel_size=5)
x = torch.tanh(conv1(x))

pool = nn.AvgPool2d(kernel_size=2, stride=2)
x = pool(x)

conv2 = nn.Conv2d(6, 16, kernel_size=5)
x = torch.tanh(conv2(x))

x = pool(x)

conv3 = nn.Conv2d(16, 120, kernel_size=5)
x = torch.tanh(conv3(x))

x_flatten = x.view(-1, 120)
fc1 = nn.Linear(120, 84)
x = torch.tanh(fc1(x_flatten))

fc2 = nn.Linear(84, 10)
x = fc2(x)



#AlexNet
x = torch.randn(1, 3, 227, 227)
conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
x = F.relu(conv1(x))

pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
x = pool1(x)

conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
x = F.relu(conv2(x))
x = pool1(x)

conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
x = F.relu(conv3(x))

conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
x = F.relu(conv4(x))

conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
x = F.relu(conv5(x))

pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
x = pool3(x)

x = x.view(x.size(0), -1)

dropout1 = nn.Dropout(p=0.5)
x = dropout1(x)

fc1 = nn.Linear(256 * 6 * 6, 4096)
x = F.relu(fc1(x))

dropout2 = nn.Dropout(p=0.5)
x = dropout2(x)

fc2 = nn.Linear(4096, 4096)
x = F.relu(fc2(x))

fc3 = nn.Linear(4096, 1000)
x = fc3(x)