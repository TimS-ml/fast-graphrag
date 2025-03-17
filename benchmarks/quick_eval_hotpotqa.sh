# https://github.com/hotpotqa/hotpot/blob/master/README.md
# https://huggingface.co/datasets/hotpotqa/hotpot_qa
# Percentage of queries with perfect retrieval: 0.803921568627451
DEBUG=2

echo -e "\n\n++++++++++++++++++++Creating databases++++++++++++++++++++"
DEBUG=$DEBUG python vdb_debug.py -n 2 -c -d hotpotqa

echo -e "\n\n++++++++++++++++++++Evaluating performance++++++++++++++++++++"
DEBUG=$DEBUG python vdb_debug.py -n 2 -b -d hotpotqa

echo -e "\n\n++++++++++++++++++++Showing results++++++++++++++++++++"
DEBUG=$DEBUG python vdb_debug.py -n 2 -s -d hotpotqa