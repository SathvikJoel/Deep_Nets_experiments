import torch
from  datetime import datetime
from pathlib import Path

def print_args(args, f_output , csv_output):

    args_dict = args.__dict__
    keys = args_dict.keys()

    #print date and time
    now = datetime.now()
    print (now.strftime("%Y-%m-%d %H:%M:%S"), file = f_output)
    print("****************************************", file=f_output)
    print("****************************************", file=f_output)
    for key in keys:
        print(f"{key}: {args_dict[key]}", file=f_output)
    print("****************************************", file=f_output)
    print("****************************************\n\n\n", file=f_output)

    print("*****************NOTES***********************", file=f_output)
    print('\n\n', file = f_output)
    print("*****************NOTES***********************", file=f_output)


#converts the time in learn.recorder.log to flaot for furthur computation
def to_int(logList):
    logList[-1] = float('.'.join(logList[-1].split(':')))
    return logList

def mean_std(log,training_time):
    log = torch.Tensor(log)
    log_mean, log_std = torch.mean(log , dim = 0), torch.std(log, dim = 0)
    training_time = torch.tensor(training_time)
    mean_training_time = torch.mean(training_time)
    std_training_time = torch.std(training_time) 
    return log_mean , log_std , mean_training_time , std_training_time

def write_stats(args, f_output, log , training_time, exe_time):
    log_mean , log_std , mean_training_time , std_training_time = mean_std(log,training_time)
    # Writing statistics to file
    print("******************STATS**********************", file=f_output)
    print("Model: ", args.model, file=f_output)
    print(f"epochs: {log_mean[0]}", file = f_output)
    print(f"Mean train loss: {log_mean[1]} +- {log_std[1]}", file=f_output)
    print(f"Mean valid loss: {log_mean[2]} +- {log_std[2]}", file=f_output)
    print(f"Mean train   accuracy: {log_mean[3]} +- {log_mean[3]}\n", file=f_output)

    print(f"Mean epoch time: {log_mean[4]} +- {log_std[4]}", file=f_output)
    print(f"Mean training time(per run): {mean_training_time} +- {std_training_time}", file=f_output)
    print(f"Total execuation time taken:{exe_time}", file = f_output)
    print("****************************************\n\n\n", file=f_output)

'''
If run == 0 , then it adds a new line to the csv output to mark the start
'''
def write_new_line(csv_output, args):
    now = datetime.now()
    sheet = Path(csv_output).open('a+')
    sheet.write(','.join(['NEW_EXP', now.strftime("%Y-%m-%d %H:%M:%S")]) + '\n')
    sheet.write(','.join(args.__dict__.keys()) + '\n')
    sheet.write(','.join([str(t) for t in args.__dict__.values()]) + '\n')

#Returns str 
def to_string(a): return str(a)