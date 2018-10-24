import os

def load_data(parts):
    target_names = ['Negative', 'Positive']
    data = []
    target = []
    if parts == 'dev':
        dirEx = os.path.join('aclImdb', 'test', 'neg')
        for filename in os.listdir(dirEx):
            with open(os.path.join(dirEx, filename), 'r') as myfile:
                data.append(myfile.read().replace('\n', ''))
        oldLen = len(data)

        target = [0] * oldLen
        dirEx = os.path.join('aclImdb', 'test', 'pos')
        for filename in os.listdir(dirEx):
            with open(os.path.join(dirEx, filename), 'r') as myfile:
                data.append(myfile.read().replace('\n', ''))
        target = target + [1] * (len(data) - oldLen)
    elif parts == 'train':
        dirEx = os.path.join('aclImdb', 'train', 'neg')
        for filename in os.listdir(dirEx):
            with open(os.path.join(dirEx, filename), 'r') as myfile:
                data.append(myfile.read().replace('\n', ''))
        oldLen = len(data)

        target = [0] * oldLen
        dirEx = os.path.join('aclImdb', 'train', 'pos')
        for filename in os.listdir(dirEx):
            with open(os.path.join(dirEx, filename), 'r') as myfile:
                data.append(myfile.read().replace('\n', ''))
        target = target + [1] * (len(data) - oldLen)
    else:
        print("Unavailable parameter name")

    return data, target, target_names
