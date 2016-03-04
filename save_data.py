import csv
import pickle

def save_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerows(data)


def dump_picle(data, filename, protocol=None):
    pickle.dump(data, open(filename, "wb"), protocol=protocol)  # protocol =4 for large file >= 4GB
    print('Pikle file has been saved.')