#Classification Problem
num_classes = 10 #num classes for classification problem
num_epochs = 5 #number of epochs for classification problem

# AI Gym Problem
resume = False # resume from previous checkpoint?
render = False
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
num_actions = 2
pong_shape = [210,160,3]

#General
batch_size = 10
conv1 = 8 #depth of first convolutional layer
conv2 = 16 #depth of second convolutional layer
FC1 = 200 #size of regression layer
lr = 0.001 #learning rate
