#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import datetime

class board_data(Dataset):
    def __init__(self, dataset):  # dataset shape: (s, p, v)
        self.states = dataset[:, 0]
        self.policies = dataset[:, 1]
        self.values = dataset[:, 2]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        state = self.states[index].transpose(2, 0, 1)
        policy = self.policies[index]
        value = self.values[index]
        return state, policy, value

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8 * 8 * 73
        self.conv_layer = nn.Conv2d(in_channels=22, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)

    def forward(self, input_state):
        input_state = input_state.view(-1, 22, 8, 8)  # Shape: (batch_size, channels, height, width)
        out = self.conv_layer(input_state)
        out = self.batch_norm(out)
        out = F.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # Value head
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_log_softmax = nn.LogSoftmax(dim=1)
        self.policy_fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)
    
    def forward(self, x):
        # Value prediction
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(-1, 8 * 8)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = torch.tanh(self.value_fc2(v))
        
        # Policy prediction
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(-1, 8 * 8 * 128)
        p = self.policy_fc(p)
        p = self.policy_log_softmax(p)
        p = torch.exp(p)
        
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv_block = ConvBlock()
        for layer_num in range(19):
            self.add_module(f"res_{layer_num}", ResBlock())
        self.output_block = OutBlock()
    
    def forward(self, x):
        out = self.conv_block(x)
        for layer_num in range(19):
            res_layer = getattr(self, f"res_{layer_num}")
            out = res_layer(out)
        p, v = self.output_block(out)
        return p, v

class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, true_value, pred_value, true_policy, pred_policy):
        value_loss = (pred_value.squeeze() - true_value) ** 2
        policy_loss = -torch.sum(true_policy.float() * torch.log(pred_policy + 1e-6), dim=1)
        total_loss = (value_loss + policy_loss).mean()
        return total_loss

def train(net, dataset, epoch_start=0, epoch_stop=20, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.train()
    
    loss_function = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)
    
    train_dataset = board_data(dataset)
    loader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    
    epoch_losses = []
    
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        running_loss = 0.0
        batch_losses = []
        
        for batch_idx, (states, policies, values) in enumerate(loader):
            states = states.float().to(device)
            policies = policies.float().to(device)
            values = values.float().to(device)
            
            optimizer.zero_grad()
            pred_policies, pred_values = net(states)
            loss = loss_function(pred_values, values, policies, pred_policies)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f'Process ID: {os.getpid()} [Epoch: {epoch + 1}, Batch: {(batch_idx + 1) * 30}/{len(train_dataset)}] '
                      f'Average Loss: {avg_loss:.3f}')
                print(f"Policy - True: {policies[0].argmax().item()}, Predicted: {pred_policies[0].argmax().item()}")
                print(f"Value - True: {values[0].item()}, Predicted: {pred_values[0].item()}")
                
                batch_losses.append(avg_loss)
                running_loss = 0.0
        
        epoch_avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_avg_loss)
        
        # Early stopping condition
        if len(epoch_losses) > 100:
            recent_avg = sum(epoch_losses[-4:-1]) / 3
            previous_avg = sum(epoch_losses[-16:-13]) / 3
            if abs(recent_avg - previous_avg) <= 0.01:
                print("Early stopping triggered.")
                break
    
    # Plotting the loss curve
    plt.figure(figsize=(8, 6))
    plt.scatter(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Epochs")
    save_path = os.path.join("./model_data/", f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png")
    plt.savefig(save_path)
    print('Training Completed and Loss Plot Saved.')
