def txt2list(filename, split_symble='\t'):
    '''Input a file formated with .txt then convert it to a list.
    And you can choose your own split symble which is defalut by \t.'''
    f = open(filename)
    if filename == None:
        print('ERROR: The file name is NONE!')
        return
    new_list = [line.strip().split(split_symble) for line in f.readlines()]
    return new_list
