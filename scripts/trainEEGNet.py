from nova2026.architecture import cnn
import torch


def max_norm_(param, max_value=1.0, eps=1e-8):
    """Applies max-norm constraint to the parameter."""
    with torch.no_grad():
        norm = param.norm(2)
        if norm > max_value:
            param.mul_(max_value / (norm + eps))


eegnet = cnn.EEGNet()
dummy_input = torch.randn(1, 64, 256)
eegnet(dummy_input)

optimizer = torch.optim.Adam(eegnet.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = eegnet(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        max_norm_(eegnet.depthwise_conv.weight, max_value=1.0)
        max_norm_(eegnet.classifier.weight, max_value=0.25)
