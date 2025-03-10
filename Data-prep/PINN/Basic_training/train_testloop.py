import time
from itertools import zip_longest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import Adam

from loss_func import loss_fn_data,pde_loss,ic_loss,boundary_loss



# check for gpu
if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)



def training_loop(epochs, model, \
                  loss_fn_data, optimizer, \
                  train_dataloader, train_loader_pde,\
                   train_loader_init,\
                  train_loader_bc_l,train_loader_bc_r,\
                      test_dataloader, pde_test_dataloader, ic_test_dataloader, \
                          test_bc_l_dataloader, test_bc_r_dataloader, temp_var):
    
    # Initialize the lists to store losses
    train_losses = []
    test_losses = []
    data_losses = []
    pde_losses = []
    ic_losses = []
    bc_losses = []
    
    data_loss_test = []
    pde_loss_test = []
    ic_loss_test = []
    bc_loss_test = []
    
    

    T_st = temp_var["T_st"]
    T_lt = temp_var["T_lt"]
    t_surrt = temp_var["t_surrt"]
    temp_init_t = temp_var["temp_init_t"]
    
    model.to(device)  # Move the model to the GPU
    
    for epoch in range(epochs):
        
        model.train()  # Set the model to training mode
        train_loss = 0  # Initialize the training loss for this epoch
        data_loss_b = 0  # Data loss accumulator
        phy_loss_acc = 0  # PDE loss accumulator
        init_loss_acc = 0  # Initial condition loss accumulator
        bc_loss_acc = 0  # Boundary condition loss accumulator
        
        
        data_loss_t = 0  # Data loss accumulator
        phy_loss_t = 0
        ic_loss_t = 0
        bc_l_loss_t = 0
        bc_r_loss_t = 0


        # Loop through the training data loaders
        for (batch, batch_pde, batch_init, batch_left, batch_right) in \
             zip_longest(train_dataloader, train_loader_pde, train_loader_init, train_loader_bc_l, train_loader_bc_r):
            
            # print(len(train_dataloader))
            # print(len(train_loader_pde))
            # print(len(train_loader_init))
            # print(len(train_loader_bc_l))
            # print(len(train_loader_bc_r))
            if batch is None or batch_pde is None or batch_init is None or batch_left is None or batch_right is None:
                continue  # Skip this iteration
            # Extract inputs from each batch
            inputs, temp_inp = batch
            inputs_pde = batch_pde
            inputs_init = batch_init
            inputs_left = batch_left
            inputs_right = batch_right

            inputs, temp_inp = inputs, temp_inp
            inputs_pde = inputs_pde
            inputs_init = inputs_init
            inputs_left = inputs_left
            inputs_right = inputs_right
            
            
            optimizer.zero_grad()  # Zero the gradients before backpropagation
            
            # Forward pass for data prediction
            u_pred_d = model(inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1))
            data_loss = loss_fn_data(u_pred_d, temp_inp)  # Data loss
            
            # Forward pass for initial condition prediction
            u_initl = model(inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1))
            init_loss = ic_loss(model,inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1),temp_init_t)  # Initial condition loss
            
            # Forward pass for boundary conditions
            u_left = model(inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1))
            u_right = model(inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1))
            
            # Boundary condition loss (left and right)
            bc_loss_left = boundary_loss(model, inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1), t_surrt,temp_init_t)
            bc_loss_right = boundary_loss(model, inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1), t_surrt,temp_init_t)
            bc_loss = 0.5*(bc_loss_left + bc_loss_right)
            # Calculate individual losses
           
            phy_loss = pde_loss(model, inputs_pde[:, 0].unsqueeze(1), inputs_pde[:, 1].unsqueeze(1), T_st, T_lt)  # PDE loss
          
          
            # Define weights for the different losses
            w0, w1, w2, w3 = 1, 1, 1,1
            # Calculate total loss
            loss =  w1 * phy_loss + w2 * init_loss + w3 * bc_loss
            # loss =  w1 * phy_loss + w2 * init_loss + w3 * bc_loss
            # Backpropagation
            loss.backward(retain_graph=True)  # Backpropagate the gradients
            
            def closure():
                optimizer.zero_grad()
                loss.backward()
                return loss
            
            if optimizer.__class__ == torch.optim.Adam:
                optimizer.step()  # Update the weights
            else:
                optimizer.step(closure)
             
            # Accumulate losses for tracking
            train_loss += loss.item()
            
            data_loss_b += data_loss.item()
            phy_loss_acc += phy_loss.item()
            init_loss_acc += init_loss.item()
            bc_loss_acc += bc_loss.item()
        
          
        
        # Append losses to respective lists for tracking
        if len(train_dataloader) > 0:
            train_losses.append(train_loss / len(train_dataloader))
            data_losses.append(data_loss_b / len(train_dataloader))
        if len(train_loader_pde) > 0:
            pde_losses.append(phy_loss_acc / len(train_loader_pde))
        if len(train_loader_init) > 0:
            ic_losses.append(init_loss_acc / len(train_loader_init))
        if len(train_loader_bc_l) > 0:
            bc_losses.append(bc_loss_acc / len(train_loader_bc_l))
        
       

        # Set model to evaluation mode for testing
        model.eval()
        test_loss = 0
        
        # Evaluate on test data without gradient calculation
        for (batch, batch_pde, batch_init, batch_left, batch_right) in zip_longest(test_dataloader, \
            pde_test_dataloader, ic_test_dataloader, test_bc_l_dataloader, test_bc_r_dataloader):
            
            if batch is None or batch_pde is None or batch_init is None or batch_left is None or batch_right is None:
               continue  # Skip this iteration
            inputs, temp_inp = batch
            inputs_pde = batch_pde
            inputs_init = batch_init
            inputs_left = batch_left
            inputs_right = batch_right
            
            inputs, temp_inp = inputs, temp_inp
            inputs_pde = inputs_pde
            inputs_init = inputs_init
            inputs_left = inputs_left
            inputs_right = inputs_right
            
            
            u_pred = model(inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1))
            data_loss_t = loss_fn_data(u_pred, temp_inp)
            
            u_initl = model(inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1))
            init_loss_t = ic_loss(model,inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1),temp_init_t)
            
            u_left = model(inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1))
            u_right = model(inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1))
            
            bc_loss_left_t = boundary_loss(model, inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1),t_surrt,temp_init_t)
            bc_loss_right_t = boundary_loss(model, inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1),t_surrt,temp_init_t)
            bc_loss_t = 0.5*(bc_loss_left_t + bc_loss_right_t)
            
            phy_loss_t = pde_loss(model, inputs_pde[:, 0].unsqueeze(1), inputs_pde[:, 1].unsqueeze(1), T_st, T_lt)
            
            w0, w1, w2, w3 = 1,1,1,1
            loss_t =  w1 * phy_loss_t + w2 * init_loss_t + w3 * bc_loss_t
            # loss_t = w1 * phy_loss_t + w2 * init_loss_t + w3 * bc_loss_t
            
            test_loss += loss_t.item()
            data_loss_t += data_loss_t.item()
            phy_loss_t += phy_loss_t.item()
            ic_loss_t += init_loss_t.item()
            bc_l_loss_t += bc_loss_t.item()
        
        # Normalize the test loss by the number of test batches
        if len(test_dataloader) > 0:
            test_losses.append(test_loss / len(test_dataloader))
            data_loss_test.append(data_loss_t / len(test_dataloader))
        if len(pde_test_dataloader) > 0:
            pde_loss_test.append(phy_loss_t / len(pde_test_dataloader))
        if len(ic_test_dataloader) > 0:
            ic_loss_test.append(ic_loss_t / len(ic_test_dataloader))
        if len(test_bc_l_dataloader) > 0:
            bc_loss_test.append(bc_loss_t / len(test_bc_l_dataloader))
        

         # Saving the best model in the training loop
        if epoch == 0:
            best_loss = test_loss / len(test_dataloader)
            best_model = model
        else:
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model
        # Empty CUDA cache to free memory
        torch.cuda.empty_cache()

        # Get the train losses in an array  format
        loss_train = {"train-loss": train_losses, "data-loss": data_losses, "pde-loss": pde_losses, "ic-loss": ic_losses, "bc-loss": bc_losses}
        
        loss_test = {"test-loss": test_losses, "data-loss": data_loss_test, "pde-loss": pde_loss_test, "ic-loss": ic_loss_test, "bc-loss": bc_loss_test}

        # save this losses to a csv file

        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f" ")
            print(f"--"*50)
            print(f"| Epoch {epoch+1},            | Training-Loss {train_loss:.4e},| Test-Loss {test_loss:.4e}   |")
            print(f"--"*50)
            print(f"| Data-loss {data_loss:.4e},| pde-loss {phy_loss_acc:.4e},| initc-loss {init_loss:.4e},|bc_loss {bc_loss:.4e}|") 
            print(f"--"*50)
            print(f"| Data-loss-test {data_loss_t:.4e},| pde-loss-test {phy_loss_t:.4e},| initc-loss-test {init_loss_t:.4e},|bc_loss-test {bc_loss_t:.4e}|")
            print(f"--"*50)
            print(f" ")

        if epoch == (epochs-1):
            print(f" ")
            print(f"--"*50)
            print(f"| Epoch {epoch+1},            | Training-Loss {train_loss:.4e},| Test-Loss {test_loss:.4e}   |")
            print(f"--"*50)
            print(f"| Data-loss {data_loss:.4e},| pde-loss {phy_loss_acc:.4e},| initc-loss {init_loss:.4e},|bc_loss {bc_loss:.4e}|") 
            print(f"--"*50)
            print(f"| Data-loss-test {data_loss_t:.4e},| pde-loss-test {phy_loss_t:.4e},| initc-loss-test {init_loss_t:.4e},|bc_loss-test {bc_loss_t:.4e}|")
            print(f"--"*50)
            print(f" ")
    # Return all collected losses for further analysis
    # return train_losses, test_losses, pde_losses, bc_losses, ic_losses, data_losses
    return loss_train, loss_test, best_model


