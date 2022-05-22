import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

def load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data


def dump_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
        

def setup_logging(log_dir, log_level, trace_id):
    log_cfg = {
        "level": log_level,
        "format": f"$asctime|$process|$name|$filename:$lineno|$levelname|] $message",
        "style": "$",
    }
    if log_dir:
        log_fn = Path(log_dir)
        log_cfg.update(
            {
                "filename": str(log_fn / f"{trace_id}.log"),
                "filemode": "a",
            }
        )
        log_fn.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(**log_cfg)
    logging.captureWarnings(True)
 
def get_current_logger():
    return logging.getLogger()   

  

def plot_multi_test_out(predicted_datas, real_values, interval, model_tag,plot_out_dir):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(real_values, label='True Data')
    
    for i,predicted_data in enumerate(predicted_datas):
        padding = [None for j in range(interval*i+len(predicted_data))]
    
        base_value = real_values[i * interval]
        
        
        adjust_predict = [(i+1)*base_value for i in predicted_data]
        plt.plot(padding + adjust_predict, label='Prediction')
    
    plt.savefig("{}/{}.png".format(plot_out_dir,model_tag))
    

    
def plot_results_real_multiple_dense(predicted_data,real_values,interval,model_tag):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(real_values, label='True Data')
    for i, data in enumerate(predicted_data):
        
        padding = [None for p1 in range((i) * interval)] + [None for p2 in data]
        base_value_index = interval * i
        base_value = real_values[base_value_index]
        adjust_data = [(i+1)*base_value for i in data]
        plt.plot(padding + adjust_data, label='Prediction')
    plt.savefig("log/{}.png".format(model_tag))

def plot_results_multiple(predicted_data, true_data, seq_len,model_tag):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range((i) * seq_len)]
        plt.plot(padding + data, label='Prediction')
        # plt.legend()
    plt.savefig("log/{}.png".format(model_tag))
    
def plot_results_point_by_point(predicted_data, true_data ,seq_len, model_tag):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig("log/{}.png".format(model_tag))
    
def plot_points(values,model_tag):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

	# Pad the list of predictions to shift it in the graph to it's correct start
    plt.plot(values, label='real valuse')
    plt.legend()
    plt.savefig("log/{}.png".format(model_tag))