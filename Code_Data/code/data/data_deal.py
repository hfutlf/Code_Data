def data_gettype(filename):
    with open(filename) as file_object:
        lines = file_object.readlines()
    data_lines = []
    for i in range(len(lines)):
        if i % 3 == 1:
            data_lines.append(lines[i].strip().split('\t'))
    data_lines = sum(data_lines, [])
    return data_lines


def get_kc_set(path):
    training_path = path + '/training.txt'
    testing_path = path + '/testing.txt'
    train_typeset = data_gettype(training_path)
    test_typeset = data_gettype(testing_path)
    type_set = train_typeset + test_typeset
    temp = {}
    temp = temp.fromkeys(type_set)
    type_set = list(temp.keys())
    type_length = len(type_set)
    type_trans = dict()
    for i in range(type_length):
        type_trans[int(type_set[i])] = i
    print(type_trans)
    return type_trans, type_length, train_typeset, test_typeset
