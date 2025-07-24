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
    print('do this: separate finish all extract_feature_mentions and cache')
    print('https://chatgpt.com/g/g-p-686d46da9dd0819180e8ab3240a7ee58-new-llm-market/c/6882a278-a874-8332-b8c9-bd94748ab513')
    input('wait')
    main()
