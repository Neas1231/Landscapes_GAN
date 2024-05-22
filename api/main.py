import os.path

import cv2
import streamlit as st
import torch
from torch import nn
from glob import glob

st.header('Landscapes GAN')


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, stride=1, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True), )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True), )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True), )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True), )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, stride=2, kernel_size=4, padding=1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 3, 64, 64)
        return x


weights = list(map(lambda x: os.path.basename(x),glob('./weights/Generator*.pth')))
weights.sort(key=lambda f: int(f[f.rfind('_')+1:f.rfind('.')]))
model_weights = st.selectbox(
    "Select model weights",
    weights,
    index=None,
    placeholder="Select weights...",
)
if model_weights:
    generate = st.button('Generate')
    model = torch.load('./weights/'+model_weights, map_location='cpu')
    model.eval()
    if generate:
        img = (((model((torch.rand(1, 100) - 0.5) / 0.5)[0] + 1) / 2).clamp(0, 1)).detach().numpy().transpose(1, 2, 0)
        img = cv2.resize(img, (320, 300))
        st.image(img)
