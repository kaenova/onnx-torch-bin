import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import random
import numpy as np

from modules.model import create_model
from modules.data import data_generator

SEED_NUM = 2022
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

DEVICE = 'cpu'

MODEL_NAME="test1"
NUM_EPOCH = 10000

if __name__ == "__main__":
    # Try create model
    model = create_model()
    
    # Get data and preprocess
    data = data_generator()
    x = data['input']
    y = data['output']
    
    # Change to batch_num, data
    x = np.reshape(x,(len(x), 1))
    y = np.reshape(y, (len(y), 1))
    
    # Split
    x_train = x[:80]
    x_test = x[80:]
    y_train = y[:80]
    y_test = y[80:]

    # Change to tensors 
    x_train = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)    
    x_test = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)   
    y_train = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)    
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DEVICE) 
    
    # Try to inferece
    with torch.no_grad():
        y_pred = model(x_train)
        
    # Loss criterion
    criterion = nn.MSELoss()
    # Optimier
    optimizer = optim.Adam(model.parameters(), lr=0.003)
        
    # Training loop
    with trange(NUM_EPOCH) as t:   
        for i in t:            
            
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            # Update visualization on loss
            if i % 500 == 0:
                t.set_postfix(loss=loss.item())
                
    # Predict
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = nn.functional.mse_loss(y_pred, y_test)
        print(f"Test loss:", test_loss.item())
        
    # Save model
    torch.save(model, f"model/{MODEL_NAME}.pt")
