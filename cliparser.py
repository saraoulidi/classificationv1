import argparse

parser = argparse.ArgumentParser(prog='train', description='This will help you to train your model')
parser.add_argument('model_noun', type=str, help='The noun of the model that you want !!')
parser.add_argument('-b', '--batch_size', nargs='?', type=int, default=128,
                    help='The number of samples that will be propagated through the network')
parser.add_argument('-ep', '--epochs', nargs='?', type=int, default=10, help='The one pass over the entire dataset')
parser.add_argument('-p', '--path', nargs='?', type=str, default='./Dataset', help='The path to the Dataset')
parser.add_argument('-d1', '--dropout1', nargs='?', type=float, default=0.2, help='the number of dropout')
parser.add_argument('-d2', '--dropout2', nargs='?', type=float, default=0.2, help='the number of dropout')
parser.add_argument('-o', '--optimizer', nargs='?', type=str, default='adam', help='the optimizer')
parser.add_argument('-l', '--learningrate', nargs='?', type=float, default=0.001,
                    help='the learning_rate parameter for the optimizer')
parser.add_argument('-m', '--momentum', nargs='?', type=float, default=0.2, help='the momentum parameter')
parser.add_argument('-r', '--rho', nargs='?', type=float, default=0.0, help='the rho parameter')
parser.add_argument('-b1', '--beta1', nargs='?', type=float, default=0.9, help='the beta_1 parameter')
parser.add_argument('-b2', '--beta2', nargs='?', type=float, default=0.9, help='the beta_2 parameter')
parser.add_argument('--nesterov', action='store_false', help='the nesterov parameter')
parser.add_argument('--amsgrad', action='store_false', help='the amsgrad parameter')
parser.add_argument('-K', '--Kfold', nargs='?', type=int, default=3, help='The number of folds')
parser.add_argument('--tflite', action='store_true', help='If you want to use tf lite')


args = parser.parse_args()
n_split=args.Kfold
epochs = args.epochs
model_nom = args.model_noun
pathData = args.path
optimizer = args.optimizer
dropout1 = args.dropout1
dropout2 = args.dropout2
batch_size = args.batch_size
learningrate = args.learningrate
momentum = args.momentum
nesterov = args.nesterov
beta1 = args.beta1
beta2 = args.beta2
amsgrad = args.amsgrad
tflite = args.tflite