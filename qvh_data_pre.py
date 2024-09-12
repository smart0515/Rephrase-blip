import os
import json
import pandas as pd
from glob import glob
# from tqdm import tqdm
# tqdm for notebooks
from tqdm import tqdm_notebook as tqdm
import random


def save_json(content, save_path):
    # if no such directory, create one
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
    
# ann_root = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/annotations/QVHighlights'
ann_root = "/home/jejekim/VTimeLLM/data_qvh"
train_path = ann_root + '/highlight_train_release.jsonl'
val_path = ann_root + '/highlight_val_release.jsonl'
test_path = ann_root + '/highlight_test_release.jsonl'

train = load_jsonl(train_path)
val = load_jsonl(val_path)
test = load_jsonl(test_path)

def process_QVH(data, relative_time=False, save_float=False, is_test=False):
    out = []
    for d in data:
        sample = {}
        sample['video'] = d['vid']
        sample['qid'] = 'QVHighlight_' + str(d['qid'])
        sample['query'] = d['query']
        duration = d['duration']
        sample['duration'] = duration

        if not is_test:
            windows = d['relevant_windows']
            if relative_time:
                relative_time_windows = []
                for window in windows:
                    start = window[0] / duration
                    end = window[1] / duration

                    if save_float:
                        relative_time_windows.append([round(start, 2), round(end, 2)])
                    else:
                        relative_time_windows.append([int(round(start, 2) * 100), int(round(end, 2) * 100)])
                sample['relevant_windows'] = relative_time_windows
            else:
                sample['relevant_windows'] = windows
        else:
            sample['relevant_windows'] = [[0, 150]] # dummy value

        out.append(sample)

    return out

save_float = False
relative_time = False

new_train = process_QVH(train, relative_time=relative_time, save_float=save_float)
new_val = process_QVH(val, relative_time=relative_time, save_float=save_float)
new_test = process_QVH(test, relative_time=relative_time, save_float=save_float, is_test=True)

# save data
if save_float and relative_time:
    save_json(new_train, ann_root + '/lavis/train_relative_float.json')
    save_json(new_val, ann_root + '/lavis/val_relative_float.json')
    save_json(new_test, ann_root + '/lavis/test_relative_float_dummy.json')
elif save_float and not relative_time:
    save_json(new_train, ann_root + '/lavis/train_float.json')
    save_json(new_val, ann_root + '/lavis/val_float.json')
    save_json(new_test, ann_root + '/lavis/test_float_dummy.json')
elif not save_float and relative_time:
    save_json(new_train, ann_root + '/lavis/train_relative.json')
    save_json(new_val, ann_root + '/lavis/val_relative.json')
    save_json(new_test, ann_root + '/lavis/test_relative_dummy.json')
else:
    save_json(new_train, ann_root + '/lavis/train.json')
    save_json(new_val, ann_root + '/lavis/val.json')
    save_json(new_test, ann_root + '/lavis/test_dummy.json')