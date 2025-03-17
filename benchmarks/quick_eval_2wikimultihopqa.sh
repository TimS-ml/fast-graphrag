# https://github.com/Alab-NII/2wikimultihop
# https://www.dropbox.com/scl/fi/aasqsj45yokx71pnm8ctr/data_ids.zip?dl=0&e=1&file_subpath=/data_ids&rlkey=72n2p6jywhfmm6kdeuzz8c55u
# https://huggingface.co/datasets/scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35
# Percentage of queries with perfect retrieval: 0.5098039215686274
# [multihop] Percentage of queries with perfect retrieval: 0.375

# Improved HNSW
# Percentage of queries with perfect retrieval: 0.47058823529411764
# [multihop] Percentage of queries with perfect retrieval: 0.375

DEBUG=2
# SCRIPT_NAME=vdb_debug.py
SCRIPT_NAME=vdb_benchmark_v2.py
# N=2
N=51

echo -e "\n\n++++++++++++++++++++Creating databases++++++++++++++++++++"
DEBUG=$DEBUG python $SCRIPT_NAME -n $N -c -d 2wikimultihopqa

echo -e "\n\n++++++++++++++++++++Evaluating performance++++++++++++++++++++"
DEBUG=$DEBUG python $SCRIPT_NAME -n $N -b -d 2wikimultihopqa

echo -e "\n\n++++++++++++++++++++Showing results++++++++++++++++++++"
DEBUG=$DEBUG python $SCRIPT_NAME -n $N -s -d 2wikimultihopqa