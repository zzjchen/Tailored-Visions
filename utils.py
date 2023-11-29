import jsonlines
import os

def make_path(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def print_args(*argv, **kwargs):
    for arg in argv:
        print(f"{arg}")
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))

def printspace(strings):
    print()
    print(strings)
    print()

'''jsonlines utils'''
def read_jsonl_data(path_and_name):
    data = []
    with open(path_and_name, 'r', encoding="utf-8") as fd:
        for l in jsonlines.Reader(fd):
            data.append(l)
    return data

def write_jsonl_data(items, path_and_name):
    with jsonlines.open(path_and_name, 'w') as writer:
        writer.write_all(items)

def read_multi_jsonl_file(lst_file):
    '''
    :param lst_file: list of jsonl file
    :return: merged jsonl list
    '''
    datas = []
    for i in lst_file:
        with open(i ,'r',encoding="utf-8") as fd:
            for l in jsonlines.Reader(fd):
                datas.append(l)
    return datas

def read_txt_data(path_and_name):
    with open(path_and_name) as f:
        lines = f.readlines()
    return lines

if __name__ == "__main__":
    pass
