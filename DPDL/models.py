import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stat


class dualCNN(nn.Module):
    def __init__(self, nb_classes,input_size,input_channel,dropout_rate=0):
        super(dualCNN, self).__init__()
        if (dropout_rate<0) or (dropout_rate>1): dropout_rate =0
        self.nb_classes = nb_classes
        self.input_size = input_size # Needs to be square (x,x) needs to be > 256
        self.input_channel = input_channel
        self.dropout_rate = dropout_rate

        self.S1 = nn.Sequential(
            nn.Conv2d(input_channel,16,13,1,6),     #--> (16,x,x)
            nn.ELU(),
            nn.Conv2d(16,16,13,1,6),                #--> (16,x,x)
            nn.ELU(),
            nn.AvgPool2d(4),                        #--> (16,x/4,x/4)
            nn.Conv2d(16,32,13,1,6),                #--> (32,x/4,x/4)
            nn.ELU(),
            nn.Conv2d(32,32,13,1,6),                #--> (32,x/4,x/4)
            nn.ELU(),
            nn.AvgPool2d(4),                        #--> (32,x/16,x/16)
            nn.Conv2d(32,64,13,1,6),                #--> (64,x/16,x/16)
            nn.ELU(),
            nn.Conv2d(64,64,13,1,6),                #--> (64,x/16,x/16)
            nn.ELU(),
            nn.AvgPool2d(4),                        #--> (64,x/64,x/64)
        )
        self.S2 = nn.Sequential(
            nn.Conv2d(input_channel,16,3,1,1),     #--> (16,x,x)
            nn.ELU(),
            nn.Conv2d(16,16,3,1,1),                #--> (16,x,x)
            nn.ELU(),
            nn.MaxPool2d(4),                        #--> (16,x/4,x/4)
            nn.Conv2d(16,32,3,1,1),                #--> (32,x/4,x/4)
            nn.ELU(),
            nn.Conv2d(32,32,3,1,1),                #--> (32,x/4,x/4)
            nn.ELU(),
            nn.MaxPool2d(4),                        #--> (32,x/16,x/16)
            nn.Conv2d(32,64,3,1,1),                #--> (64,x/16,x/16)
            nn.ELU(),
            nn.Conv2d(64,64,3,1,1),                #--> (64,x/16,x/16)
            nn.ELU(),
            nn.MaxPool2d(4),                        #--> (64,x/64,x/64)
        )

        self.L1 = nn.Sequential(
            nn.Linear(int(input_size/64*input_size/64*64),int(input_size/64*input_size/64*64/2)),
            nn.BatchNorm1d(int(input_size/64*input_size/64*64/2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.L2 = nn.Sequential(
            nn.Linear(int(input_size/64*input_size/64*64),int(input_size/64*input_size/64*64/2)),
            nn.BatchNorm1d(int(input_size/64*input_size/64*64/2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.FC = nn.Sequential(
            nn.Linear(int(input_size/64*input_size/64*64),int(input_size/64*input_size/64*64/4)),
            nn.BatchNorm1d(int(input_size/64*input_size/64*64/4)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(input_size/64*input_size/64*64/4),nb_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.S1(x)
        x2 = self.S2(x)

        x1 = torch.flatten(x1, 1)
        x1 = self.L1(x1)
        x2 = torch.flatten(x2, 1)
        x2 = self.L2(x2)

        x3 = torch.cat((x1,x2),dim=1)
        x3 = x3.view(x3.size(0), -1)
        output = self.FC(x3)
        return output
