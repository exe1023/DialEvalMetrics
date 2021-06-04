import os
import sys

def modify_config_data(init_embd_file, train_batch_size, pickle_data_dir, num_train_data, \
    config_data_file, max_train_bert_epoch):
    # Modify the data configuration file
    config_data_exists = os.path.isfile(config_data_file)
    if config_data_exists:
        with open(config_data_file, 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('init_embd_file =') or
                        line.startswith('train_batch_size =') or
                        line.startswith('pickle_data_dir =') or
                        line.startswith('num_train_data =') or
                        line.startswith('max_train_bert_epoch =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            init_embd_file="'"+init_embd_file+"'"
            filedata_lines.insert(
                insert_idx, f'{"init_embd_file"} = {init_embd_file}')
            filedata_lines.insert(
                insert_idx, f'{"train_batch_size"} = {train_batch_size}')
            pickle_data_dir="'"+pickle_data_dir+"'"
            filedata_lines.insert(
                insert_idx, f'{"pickle_data_dir"} = {pickle_data_dir}')
            filedata_lines.insert(
                insert_idx, f'{"num_train_data"} = {num_train_data}')
            filedata_lines.insert(
                insert_idx, f'{"max_train_bert_epoch"} = {max_train_bert_epoch}')

        with open(config_data_file, 'w') as file:
            file.write('\n'.join(filedata_lines))
        print("{} has been updated".format(config_data_file))
    else:
        print("{} cannot be found".format(config_data_file))

    print("Data preparation finished")


if __name__ == "__main__":
    init_embd_file=sys.argv[1]
    train_batch_size=sys.argv[2]
    pickle_data_dir=sys.argv[3]
    config_data_file=sys.argv[4]
    max_train_bert_epoch=int(sys.argv[5])
    num_train_data=0

    file=os.path.join(pickle_data_dir, 'train', 'pair-1', 'original_dialog.text')
    with open(file, 'r') as f:
        for line in f.readlines():
            num_train_data +=1
    modify_config_data(init_embd_file, train_batch_size, pickle_data_dir, num_train_data, config_data_file, max_train_bert_epoch)