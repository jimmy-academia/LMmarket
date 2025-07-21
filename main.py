## python main.py --sample --dset yelp

import argparse
from dataset import make

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help="Use a small sample dataset")
    parser.add_argument('--dset', type=str, default='yelp')
    args = parser.parse_args()
    
    print("Hello from lmmarket!")
    make.main(args)
    


if __name__ == "__main__":
    main()
