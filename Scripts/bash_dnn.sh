nohup python solubility_model.py -j Modeling -m dnn -dscr rdkit_2d -s random -g 1  > dnn_rdkit_2d_random &
nohup python solubility_model.py -j Modeling -m dnn -dscr rdkit_2d -s clustering -g 0  > dnn_rdkit_2d_clustering &

