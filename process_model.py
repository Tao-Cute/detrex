import torch
import argparse
import collections

def main():
    parser = argparse.ArgumentParser(
                description='get model path')
    parser.add_argument('path', help='cocoC results path')
    args = parser.parse_args()
    new_model = collections.OrderedDict()

    model = torch.load(args.path)['state_dict']

    for key in model.keys():
        word = key.split('.')[1]
        if word == "backbone":
            new_model[key[17:]] = model[key]
            print(key[17:])

    torch.save(new_model, args.path)

if __name__ == "__main__":
    main()

