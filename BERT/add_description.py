import sys
import pickle

if len(sys.argv) < 3:
    print('Not enough arguments')
    print('add_description.py [output_path] [line1] ... [line n]')
    sys.exit()

with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

descr = []
for line in sys.argv[2:]:
    descr.append(line)
data = list(data)
data.append(descr)
data = tuple(data)

with open(sys.argv[1], 'wb') as f:
    pickle.dump(data, f)