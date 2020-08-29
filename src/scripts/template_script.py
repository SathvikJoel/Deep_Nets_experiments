'''
This is meant to be a template for all the test scripts
'''
import sys
import os
import pathlib
sys.path.insert(1, '.') 
sys.path.insert(2,'src\models\synergy_1')        
from argparse import ArgumentParser , SUPPRESS
import re
import time
import pickle
from  datetime import datetime
from fastai import *
from fastai.vision.all import *
#from fastai import *
from csv import writer
from src.lib.test_lib import *
from src.models.synergy_1.local_attention import *
import pathlib
'''
Function to parse the input
'''
def parse_input():
    
    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    # Add back help 
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='Enter all required fields'
    )
    required.add_argument('-d', '--dataset', type=str, metavar='STRING', choices=('IMAGENETTE', 'IMAGEWOOF'), help='dataset, choices {%(choices)s}', required=True)
    required.add_argument('-m', '--model_name', type=str, metavar='STRING', help='model name', required= True)
    required.add_argument('-e', '--num_epochs', type=int, metavar='NUMBER', help='number of training epochs', required= True)
    required.add_argument('-l', '--learning_rate', type=float, metavar='NUMBER', help='base learning rate',required= True)
    required.add_argument('-c', '--cortex', type = int , default = 20 , metavar='NUMBER', help = 'synergy number', required=True)
    required.add_argument('-lab', '--laboratory', type = str, metavar='STRING', help='Experiment Number string',required=True)

    optional.add_argument('-s', '--size', type = int , metavar = 'STRING', default = 128 , help = 'Size of image(default:128')
    optional.add_argument('-r', '--repetitions', type=int, metavar='NUMBER', default=1, help='number of repetitions')
    required.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default:128)')
    args = parser.parse_args()

    return args


'''
Creates required directories and files
'''
def setup():
    args = parse_input()
    path = os.getcwd()      
    #path to lab_xx folder
    out_path = os.path.join(path , 'results', 'coretex_'+ str(args.cortex), 'lab_' + str(args.laboratory))

    filepath= os.path.join(out_path, args.dataset, args.model_name, os.path.sep)

    #create folder corresponding to modle_name and dataset
    if (not os.path.exists(filepath)) :
        print(f"Generating folder {filepath}")
        os.makedirs(filepath)
    
    #output file for storing summary
    f_output = open('/home/ubuntu/deep_nets/Deep_Nets_experiments/results/cortex_1/laboratory_1/IMAGENETTE/ResNet50/abstarct.txt', 'a+')
    #f_output = open(filepath +args.dataset + os.path.sep + args.model_name + os.path.sep  + 'abstract' + '.txt', 'w+')
    #f_output = open(filepath + os.path.sep  + 'abstract' + '.txt', 'w+')
    #csv_output = filepath + args.dataset + os.path.sep + args.model_name + os.path.sep  + 'sheet' + '.csv'
    csv_output = '/home/ubuntu/deep_nets/Deep_Nets_experiments/results/cortex_1/laboratory_1/IMAGENETTE/ResNet50/sheet.csv'
    print_args(args, f_output, csv_output)


    return args , f_output, csv_output
    
if __name__ == "__main__":
    #configure the device
    device = torch.device('cuda',0)
    torch.cuda.set_device(device)

    args , f_output, csv_output = setup()

    m = globals()[args.model_name]

    #data_path = os.path.join(os.getcwd , 'src' + 'data' + args.dataset)
    path = untar_data(URLs.IMAGENETTE_160)
    dls = ImageDataLoaders.from_folder(path, valid='val',item_tfms=RandomResizedCrop(128, min_scale=0.35), 
    batch_tfms=Normalize())
    
    start_time = time.time()
    log = []
    training_time = []

    for run in range(args.repetitions):
        model = m()
        learn = Learner(dls, model,  loss_func= nn.CrossEntropyLoss() , metrics= accuracy,  cbs = CSVLogger(csv_output,append = True))
        if run == 0 :
            write_new_line(csv_output, args)

        learn.fit_one_cycle(args.num_epochs, max_lr =args.learning_rate)
        if run == 0 : print('      |           '.join((learn.recorder.metric_names).map(to_string)) , file = f_output)
        log.append(list(to_int(learn.recorder.log)))
        print('      |        '.join((learn.recorder.log).map(to_string)), file = f_output )
        training_time.append(time.time() - start_time)        
    
    exe_time = time.time() - start_time
    
    write_stats(args, f_output, log , training_time, exe_time)
    
    f_output.close()
 
    
    






        

