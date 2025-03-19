import os
import argparse
import shutil

def mv_centroids(args):
    centroids_dict = {}
    for root, dirs, files in os.walk("results"):
        tmp_files = []
        for file in files:
            if file == "centroids.txt" and (args.date in root):
                tmp_files.append(file)
        if len(tmp_files):
            centroids_dict[root.split(os.sep)[1]] = [root,tmp_files]
    print(centroids_dict)
    for k, i in centroids_dict.items():
        if not os.path.isdir(os.path.join(args.p2s, k)):
            os.makedirs(os.path.join(args.p2s, k))
        shutil.copy(os.path.join(i[0], i[1][0]), os.path.join(args.p2s, k, "centroids.txt"))


    
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Process some integers.')
    parse.add_argument('--date', '-d', type=str, help='date of the experiment')
    parse.add_argument('--p2s', '-p', type=str, help='path to store the preprocessed data', default="data")
    args = parse.parse_args()
    mv_centroids(args)