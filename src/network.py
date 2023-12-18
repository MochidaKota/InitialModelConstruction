import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MLPClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dims, activation='ReLU', dropout=0.1, batchnorm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('activation must be ReLU or LeakyReLU.')
        
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # define input layer
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(self.activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        
            # define hidden layers  
            for i in range(1, len(hidden_dims)):
                    layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                    if batchnorm == True:
                        layers.append(nn.BatchNorm1d(hidden_dims[i]))
                    layers.append(self.activation)
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
        
            # define output layer           
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.classifier(x)
        
        return x