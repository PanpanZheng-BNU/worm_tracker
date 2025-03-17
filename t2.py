import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--p2v", type=str, help="Path to the directory containing the videos")
parse.add_argument("--p2s", type=str, help="Path to store the concatenated videos")
parse.add_argument("--pool", type=int, help="Number of processes to run", default=4)
parse.add_argument("--vis", type=bool, help="Visualize the video", default=False)

args = parse.parse_args()
print(args.p2v)