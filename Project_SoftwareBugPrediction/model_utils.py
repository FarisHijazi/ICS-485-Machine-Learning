import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt                        

from IPython.display import display, HTML

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(n_epochs, loaders, model, optimizer, criterion, save_path='model', device=None, plot=True,
          stats=None, load_ckpt=False, print_every=10, mc=True):
    """
    :param n_epochs: 
    :param loaders: 
    :param model: 
    :param optimizer: 
    :param criterion: 
    :param save_path: 
    :param device: 
    :param plot: 
    :param stats: [optional] pass this to continue a `stats` object from a previous training session
    :param load_ckpt: [optional] load model checkpoint, if true, will load the checkpoint from `save_path`
    :param mc (multi-class):
    :return: stats dict, contains info about training loss epochs and times and errors
    
    ```
    train(100, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, 'model_scratch.pt')
    ```
    """
    
    import os, time, json, datetime, traceback
    from collections import deque

    def elapsed_time(duration):
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        hours, minutes = int(hours), int(minutes)
        # return "{:0>2}h:{:0>2}m:{:05.2f}s".format(hours, minutes, seconds)
        if hours == 0 and minutes == 0:
            return "{:05.2f}s".format(seconds)
        elif hours == 0:
            return "{:0>2}m:{:05.0f}s".format(minutes, seconds)
        else:
            return "{:0>2}h:{:0>2}m".format(hours, minutes)
    
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)

    validation_label = 'val' if 'val' in loaders else 'test'
    
    # initialize tracker for minimum validation loss
    stats_json_path = os.path.join('model_checkpoints', f'{save_path}.stats_json')
    
    if type(stats) is str and os.path.exists(stats_json_path): # load it if it exists
        stats = json.load(open(stats_json_path, 'r'))
    
    if load_ckpt and os.path.exists(save_path):
        try:
            print(f'Model checkpoint found, loading checkpoint "{save_path}"... (since "load_ckpt=True")')
            model.load_state_dict(torch.load(save_path))
        except Exception as e:
            print('Error loading model:', e)
        
    if type(stats) is not dict:
        stats = {} # else empty
    
    stats_defaults = {
        'valid_loss_min': np.Inf,
        'train_loss': [],
        'valid_loss': [],
        'checkpoints': [],
        'cpt_dir': 'model_checkpoints',
        'save_path': save_path+'.pt',
    }
    stats_defaults.update(stats)
    stats.update(stats_defaults)
    stats['start_time'] = time.time()

    epoch_durations = deque([], 5)
    
    if not os.path.isdir(stats['cpt_dir']):
        os.mkdir(stats['cpt_dir'])
    
    ################
    # / end of setup
    # start the training loop
    ################
    
    for epoch in range(1, n_epochs+1):
        epoch_start_time = time.time()
        # initialize variables to monitor training and validation loss
        train_loss, valid_loss = 0.0, 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            batch_start_time = time.time()
            data, target = data.to(device), target.to(device) # move to device
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like

            if mc:
                target = target.squeeze().long()
            
            optimizer.zero_grad()
            logits = model.forward(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)).item()


        ######################
        # validate the model #
        ######################
        y_pred, y_true = [], []
        if validation_label in loaders:
            with torch.no_grad():
                model.eval()
                for batch_idx, (data, target) in enumerate(loaders[validation_label]):
                    data, target = data.to(device), target.to(device) # move to device
                    
                    if mc:
                        target = target.squeeze().long()

                    logits = model.forward(data)
                    loss = criterion(logits, target)

                    valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss)).item()
                    
                    # CALC ACCURACY
                    pred = np.round(logits.cpu().detach())
                    target = np.round(target.cpu().detach())
                    
                    y_pred.extend(pred.tolist())
                    y_true.extend(target.tolist())

                    # valid_loss += loss.item()
                # valid_loss /= len(loaders['valid']) # normalizing
        else:
            valid_loss = stats['valid_loss_min'] - 0.000001

        ################################
        # printing validation statistics
        ################################
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        expected_duration_remaining = (n_epochs - epoch + 1) * np.mean(epoch_durations)
        time_passed = time.time() - stats['start_time']

        if epoch == 1:
            later = datetime.datetime.now() + datetime.timedelta(seconds=expected_duration_remaining)
            # dd/mm/YY H:M:S
            print(f'Estimated time needed:\t{elapsed_time(expected_duration_remaining)}.'
                  f'\tCome back at around:\t{later.strftime("%d/%m/%Y %H:%M")}')

        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)

        ##################
        # save the model #
        ##################
        save_path_timestamp = '{}(vl={:.3f}_{}).pt'.format(
            save_path,
            valid_loss,
            datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S"),
        )

        # if validation loss has decreased, save model
        if valid_loss < stats['valid_loss_min'] and epoch!=1:
            torch.save(model.state_dict(), os.path.join('save', save_path+'.pt') )
            stats['checkpoints'].append(valid_loss)
            # print "saving model..."
#             print(f'\tSaving model: "{save_path}"...')
        else:
            torch.save(model.state_dict(), os.path.join(stats['cpt_dir'], save_path+'_training.pt'))
            stats['checkpoints'].append(None)


        from sklearn.metrics import accuracy_score

        delta_valid_loss = valid_loss - stats['valid_loss_min']
        
        if not print_every or epoch%print_every==0:
            # Epoch:(1/20) 01m:00050s (4.8%) 	Loss: 4.889020 	Validation Loss: 4.887 (--inf)
            print('Epoch:({}/{}) {} ({:2.1f}%) \tLoss: {:.6f} \tVLoss: {:.3f}\t accuracy: {:.2f}%'.format(
                epoch, n_epochs,
                elapsed_time(epoch_duration),
                100.0 * (time_passed) / (time_passed + expected_duration_remaining), # percentage
                train_loss,
                valid_loss,
                accuracy_score(y_true, torch.Tensor(y_pred).argmax(axis=1))*100,
                )
            )

        stats['valid_loss_min'] = min(stats['valid_loss_min'], valid_loss)
        json.dump(stats, open(stats_json_path, 'w'))

    print(f"Completed {elapsed_time(time.time() - stats['start_time'])} in with a minimum validation loss of: {stats['valid_loss_min']}")
    # plotting
    if plot:
        plt.figure(figsize=(10,10))
        plt.title(save_path)
        plt.plot(stats['train_loss'], label='Training loss')
        plt.plot(stats['valid_loss'], label=f'Validation loss ("{validation_label}")')
        plt.plot(stats['checkpoints'], 'x', label='checkpoints')
        plt.legend(frameon=False)
        plt.show()

    return stats



          

def test_majority(loaders, criterion, device='cpu'):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to device
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = torch.zeros_like(target)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss)).item()
        
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        
        total += data.size(0)

    print(' Test Loss: {:.6f}'.format(test_loss))
    print(' Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
    
    

def test_mc(loaders, model, criterion, device='cpu'):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
          
    y_pred, y_true = [], []
          

    model.to(device)
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to device
        data, target = data.to(device), target.to(device).squeeze().long()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss)).item()
        
        # convert output probabilities to predicted class
#         pred = output.argmax(axis=1).squeeze()
        pred = output.data.max(1, keepdim=True)[1]
        target_pred_view = target.data.view_as(pred)
        
        
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target_pred_view)).cpu().numpy())
        
        total += data.size(0)
        
        ## rounding and storing the values
        target = np.round(target_pred_view.cpu().detach())
        pred = np.round(pred.cpu().detach())

        y_pred.extend(pred.tolist())
        y_true.extend(target.tolist())
       

    return np.array(y_true), np.array(y_pred), test_loss
#     print(' Test Loss: {:.6f}'.format(test_loss))
#     print(' Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

