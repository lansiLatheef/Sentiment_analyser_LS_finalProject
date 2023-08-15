!pip install gradio
!pip install transformers
!pip install torch torchvision torchaudio

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Define a binary classification model based on ResNet18
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Fine-tuning loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

# Setting up Hugging Face sentiment analysis pipeline
from transformers import pipeline
classifier = pipeline("sentiment-analysis")

# Model function for Gradio
def func(utterance):
    return classifier(utterance)

# Getting Gradio library
import gradio as gr
description = "This is an AI sentiment analyzer that checks and predicts the emotions in a given utterance."

app = gr.Interface(fn=func, inputs="text", outputs="text", title="Sentiment Analyzer", description=description)
app.launch()
