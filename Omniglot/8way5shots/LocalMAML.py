import os
import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)
# Check the total number of cores available for parallel processing
workers = os.cpu_count()

# Split the batch data into 4 equal sets of distinct classes
def split_data(data, labels):
    data1 = torch.zeros((len(data),len(data[0])//4, 1, 28, 28))
    data2 = torch.zeros((len(data),len(data[0])//4, 1, 28, 28))
    data3 = torch.zeros((len(data),len(data[0])//4, 1, 28, 28))
    data4 = torch.zeros((len(data),len(data[0])//4, 1, 28, 28))
    for ind in range(len(data)):
        l1 = torch.where((labels[ind,:] < 2) & (labels[ind,:] >= 0))[0]
        l2 = torch.where((labels[ind,:] < 4) & (labels[ind,:] >= 2))[0]
        l3 = torch.where((labels[ind,:] < 6) & (labels[ind,:] >= 4))[0]
        l4 = torch.where((labels[ind,:] < 8) & (labels[ind,:] >= 6))[0]
        data1[ind,...] = data[ind,l1,...]
        data2[ind,...] = data[ind,l2,...]
        data3[ind,...] = data[ind,l3,...]
        data4[ind,...] = data[ind,l4,...]
    return data1, data2, data3, data4

# Split the batch labels into 4 equal sets of distinct classes
def split_labels(data, labels):
    data1 = torch.zeros((len(data),len(data[0])//4), dtype=int)
    data2 = torch.zeros((len(data),len(data[0])//4), dtype=int)
    data3 = torch.zeros((len(data),len(data[0])//4), dtype=int)
    data4 = torch.zeros((len(data),len(data[0])//4), dtype=int)
    for ind in range(len(data)):
        l1 = torch.where((labels[ind,:] < 2) & (labels[ind,:] >= 0))[0]
        l2 = torch.where((labels[ind,:] < 4) & (labels[ind,:] >= 2))[0]
        l3 = torch.where((labels[ind,:] < 6) & (labels[ind,:] >= 4))[0]
        l4 = torch.where((labels[ind,:] < 8) & (labels[ind,:] >= 6))[0]
        data1[ind,...] = data[ind,l1,...]
        data2[ind,...] = data[ind,l2,...]
        data3[ind,...] = data[ind,l3,...]
        data4[ind,...] = data[ind,l4,...]
    return data1, data2, data3, data4

# Train the models
def train(args):
    logger.warning('Starting to train the models.')

    # Initializations
    lr = args.step_size
    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=False,
                       test_shots=5,
                       meta_train=True,
                       download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)

   
    model1 = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model2 = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model3 = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model4 = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)

    model1.to(device=args.device)
    model2.to(device=args.device)
    model3.to(device=args.device)
    model4.to(device=args.device)

    model1.train()
    model2.train()
    model3.train()
    model4.train()

    N = args.N
    W = args.W
    local_batch_size = args.batch_size
    
    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:

        for batch_idx, batch in enumerate(pbar):
            
            model1.zero_grad()
            model2.zero_grad()
            model3.zero_grad()
            model4.zero_grad()

            train_inputs, train_targets = batch['train']         # support set
            test_inputs, test_targets = batch['test']            # query set
            
            # splitting the data and labels
            train_inputs1, train_inputs2, train_inputs3, train_inputs4 = split_data(train_inputs, train_targets)
            train_targets1, train_targets2, train_targets3, train_targets4 = split_labels(train_targets, train_targets)

            test_inputs1, test_inputs2, test_inputs3, test_inputs4 = split_data(test_inputs, test_targets)
            test_targets1, test_targets2, test_targets3, test_targets4 = split_labels(test_targets, test_targets)
            
            # data sampled from all classes (used for measuring the test accuracy of each node)
            train_inputs5 = train_inputs
            train_targets5 = train_targets
            test_inputs5 = test_inputs
            test_targets5 = test_targets
            
            # local data for node 1
            train_inputs1 = train_inputs1.to(device=args.device)
            train_targets1 = train_targets1.to(device=args.device)
            test_inputs1 = test_inputs1.to(device=args.device)
            test_targets1 = test_targets1.to(device=args.device)

            # local data for node 2
            train_inputs2 = train_inputs2.to(device=args.device)
            train_targets2 = train_targets2.to(device=args.device)
            test_inputs2 = test_inputs2.to(device=args.device)
            test_targets2 = test_targets2.to(device=args.device)

            # local data for node 3
            train_inputs3 = train_inputs3.to(device=args.device)
            train_targets3 = train_targets3.to(device=args.device)
            test_inputs3 = test_inputs3.to(device=args.device)
            test_targets3 = test_targets3.to(device=args.device)

            # local data for node 4
            train_inputs4 = train_inputs4.to(device=args.device)
            train_targets4 = train_targets4.to(device=args.device)
            test_inputs4 = test_inputs4.to(device=args.device)
            test_targets4 = test_targets4.to(device=args.device)

            # data used for measuring the test accuracy of each node
            train_inputs5 = train_inputs5.to(device=args.device)
            train_targets5 = train_targets5.to(device=args.device)
            test_inputs5 = test_inputs5.to(device=args.device)
            test_targets5 = test_targets5.to(device=args.device)

            # initialize the loss and accuracy parameters
            outer_loss1 = torch.tensor(0., device=args.device)
            accuracy1 = torch.tensor(0., device=args.device)
            outer_loss2 = torch.tensor(0., device=args.device)
            accuracy2 = torch.tensor(0., device=args.device)
            outer_loss3 = torch.tensor(0., device=args.device)
            accuracy3 = torch.tensor(0., device=args.device)
            outer_loss4 = torch.tensor(0., device=args.device)
            accuracy4 = torch.tensor(0., device=args.device)
            
            acc1 = torch.tensor(0., device=args.device)
            acc2 = torch.tensor(0., device=args.device)
            acc3 = torch.tensor(0., device=args.device)
            acc4 = torch.tensor(0., device=args.device)

            # train each model for each task in the local dataset
            for task_idx, (train_input1, train_target1, test_input1, test_target1, train_input2, train_target2, test_input2, test_target2, train_input3, train_target3, test_input3, test_target3, train_input4, train_target4, test_input4, test_target4, train_input5, train_target5, test_input5, test_target5) in enumerate(zip(train_inputs1, train_targets1, test_inputs1, test_targets1, train_inputs2, train_targets2, test_inputs2, test_targets2, train_inputs3, train_targets3, test_inputs3, test_targets3, train_inputs4, train_targets4, test_inputs4, test_targets4, train_inputs5, train_targets5, test_inputs5, test_targets5)):
  
                # forward-pass for each node
                train_logit1 = model1(train_input1)
                train_logit2 = model2(train_input2)
                train_logit3 = model3(train_input3)
                train_logit4 = model4(train_input4)
                
                # evaluate the loss 
                inner_loss1 = F.cross_entropy(train_logit1, train_target1)
                inner_loss2 = F.cross_entropy(train_logit2, train_target2)
                inner_loss3 = F.cross_entropy(train_logit3, train_target3)
                inner_loss4 = F.cross_entropy(train_logit4, train_target4)

                model1.zero_grad()
                model2.zero_grad()
                model3.zero_grad()
                model4.zero_grad()
                
                # adapt to the support set
                params1 = gradient_update_parameters(model1,
                                                    inner_loss1,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)
                params2 = gradient_update_parameters(model2,
                                                    inner_loss2,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)
                params3 = gradient_update_parameters(model3,
                                                    inner_loss3,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)
                params4 = gradient_update_parameters(model4,
                                                    inner_loss4,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)
                
                # forward-pass for each node using new initialization
                test_logit1 = model1(test_input1, params=params1)
                test_logit2 = model2(test_input2, params=params2)
                test_logit3 = model3(test_input3, params=params3)
                test_logit4 = model4(test_input4, params=params4)
                
                # evaluate the loss again
                outer_loss1 += F.cross_entropy(test_logit1, test_target1)
                outer_loss2 += F.cross_entropy(test_logit2, test_target2)
                outer_loss3 += F.cross_entropy(test_logit3, test_target3)
                outer_loss4 += F.cross_entropy(test_logit4, test_target4)
                
                # forward-pass using the data samples from each class (for evaluating the test accuracy)
                tl1 = model1(train_input5)
                tl2 = model2(train_input5)
                tl3 = model3(train_input5)
                tl4 = model4(train_input5)
                
                # evaluate the loss based on this data
                il1 = F.cross_entropy(tl1, train_target5)
                il2 = F.cross_entropy(tl2, train_target5)
                il3 = F.cross_entropy(tl3, train_target5)
                il4 = F.cross_entropy(tl4, train_target5)
                
                # try to adapt to the tasks in the support set of this data
                pt1 = gradient_update_parameters(model1,
                                                il1,
                                                step_size=args.step_size,
                                                first_order=args.first_order)
                pt2 = gradient_update_parameters(model2,
                                                il2,
                                                step_size=args.step_size,
                                                first_order=args.first_order)
                pt3 = gradient_update_parameters(model3,
                                                il3,
                                                step_size=args.step_size,
                                                first_order=args.first_order)
                pt4 = gradient_update_parameters(model4,
                                                il4,
                                                step_size=args.step_size,
                                                first_order=args.first_order)
                
                with torch.no_grad():
                    # evaluate the train accuracy based on local data
                    accuracy1 += get_accuracy(test_logit1, test_target1)
                    accuracy2 += get_accuracy(test_logit2, test_target2)
                    accuracy3 += get_accuracy(test_logit3, test_target3)
                    accuracy4 += get_accuracy(test_logit4, test_target4)
                    
                    # evaluate the test accuracy based on the data sampled from each class
                    acc1 += get_accuracy(model1(test_input5, params=pt1), test_target5)
                    acc2 += get_accuracy(model2(test_input5, params=pt2), test_target5)
                    acc3 += get_accuracy(model3(test_input5, params=pt3), test_target5)
                    acc4 += get_accuracy(model4(test_input5, params=pt4), test_target5)
                    
                        
            outer_loss1.div_(local_batch_size)
            outer_loss2.div_(local_batch_size)
            outer_loss3.div_(local_batch_size)
            outer_loss4.div_(local_batch_size)

            accuracy1.div_(local_batch_size)
            accuracy2.div_(local_batch_size)
            accuracy3.div_(local_batch_size)
            accuracy4.div_(local_batch_size)

            acc1.div_(local_batch_size)
            acc2.div_(local_batch_size)
            acc3.div_(local_batch_size)
            acc4.div_(local_batch_size)

            outer_loss1.backward()
            outer_loss2.backward()
            outer_loss3.backward()
            outer_loss4.backward()
            
            # Save the test accuracy results in .txt files
            
            file = open('results/LocalMAML_acc1.txt', 'a')
            file.write("%s\n" % float(acc1))
            file.close()
            file = open('results/LocalMAML_acc2.txt', 'a')
            file.write("%s\n" % float(acc2))
            file.close()
            file = open('results/LocalMAML_acc3.txt', 'a')
            file.write("%s\n" % float(acc3))
            file.close()
            file = open('results/LocalMAML_acc4.txt', 'a')
            file.write("%s\n" % float(acc4))
            file.close()
            
            # Save the test loss results in .txt files
            file = open('results/LocalMAML_loss1.txt', 'a')
            file.write("%s\n" % float(outer_loss1))
            file.close()
            file = open('results/LocalMAML_loss2.txt', 'a')
            file.write("%s\n" % float(outer_loss2))
            file.close()
            file = open('results/LocalMAML_loss3.txt', 'a')
            file.write("%s\n" % float(outer_loss3))
            file.close()
            file = open('results/LocalMAML_loss4.txt', 'a')
            file.write("%s\n" % float(outer_loss4))
            file.close()

            # update each local model using the weight mixing strategy
            with torch.no_grad():
                for param1, param2, param3, param4 in zip(model1.parameters(), model2.parameters(), model3.parameters(), model4.parameters()):

                    # x^{k+1} = x^{k} - lr* grad^{k}
                    param1 -= lr * param1.grad
                    param2 -= lr * param2.grad
                    param3 -= lr * param3.grad
                    param4 -= lr * param4.grad

                    pd1 = param1
                    pd2 = param2
                    pd3 = param3
                    pd4 = param4

                    # no weighted aggregation because W is an identity matrix
                    param1 *= W[0,0]
                    param1 += W[0,1]*pd2 + W[0,2]*pd3 + W[0,3]*pd4
                    param2 *= W[1,1]
                    param2 += W[1,0]*pd1 + W[1,2]*pd3 + W[1,3]*pd4
                    param3 *= W[2,2]
                    param3 += W[2,0]*pd1 + W[2,1]*pd2 + W[2,3]*pd4
                    param4 *= W[3,3]
                    param4 += W[3,0]*pd1 + W[3,1]*pd2 + W[3,2]*pd3                
                
            pbar.set_postfix(acc1='{0:.4f}'.format(acc1.item()),
                            acc2='{0:.4f}'.format(acc2.item()),
                            acc3='{0:.4f}'.format(acc3.item()),
                            acc4='{0:.4f}'.format(acc4.item()))
                

            if batch_idx >= args.num_batches:
                break
                
class Object(object):
    pass


N = 4                                                    # Number of nodes of the 
W = torch.eye(N)                                         # Independent training

# Set the parameters
parser = Object()
parser.folder = '/Code/data/'                            # Set the path
parser.num_shots = 5                                     # Set the number of shots per class
parser.num_ways= 8                                       # Set the number of ways/classes per task
parser.shuffle=True
parser.download=True
parser.device = 'cpu'                                    # Set the device you want to train on
parser.N = N                                     
parser.W = W                                     
parser.batch_size = 32                                   # Set the batch-size
parser.num_batches = 60000                               # Set maximum number of batches

parser.num_workers = int(workers/2)                      # Use half the total number of cores to train the model
parser.hidden_size = 64                                  # Set the size of the hidden layers
parser.step_size = 1e-2                                  # Set the step-size/learning rate
parser.first_order = True                                # Set to True for first-order approximation (faster) and False (best results) otherwise 
parser.output_folder = parser.folder

# Train the models
train(parser)                                     