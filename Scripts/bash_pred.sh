time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_one_72_cpds.csv -o ./../Test/prediction/logS_testset_one_72_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_one_72_pred.csv -o ./../Test/prediction/logS_testset_one_72_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_one_72_pred.csv -o ./../Test/prediction/logS_testset_one_72_pred.csv -c rdkit_ClogP

time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_two_132_cpds.csv -o ./../Test/prediction/logS_testset_two_132_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_two_132_pred.csv -o ./../Test/prediction/logS_testset_two_132_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_two_132_pred.csv -o ./../Test/prediction/logS_testset_two_132_pred.csv -c rdkit_ClogP

time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_three_148_cpds.csv -o ./../Test/prediction/logS_testset_three_148_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_three_148_pred.csv -o ./../Test/prediction/logS_testset_three_148_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_three_148_pred.csv -o ./../Test/prediction/logS_testset_three_148_pred.csv -c rdkit_ClogP

time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_four_900_cpds.csv -o ./../Test/prediction/logS_testset_four_900_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_four_900_pred.csv -o ./../Test/prediction/logS_testset_four_900_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_four_900_pred.csv -o ./../Test/prediction/logS_testset_four_900_pred.csv -c rdkit_ClogP

time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_five_8613_cpds.csv -o ./../Test/prediction/logS_testset_five_8613_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_five_8613_pred.csv -o ./../Test/prediction/logS_testset_five_8613_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_five_8613_pred.csv -o ./../Test/prediction/logS_testset_five_8613_pred.csv -c rdkit_ClogP

time python solubility_model.py -j Prediction -m random_forest -dscr rdkit_2d -s clustering -i ./../Test/clean/logS_testset_six_21_cpds.csv -o ./../Test/prediction/logS_testset_six_21_pred.csv -c False
time python solubility_model.py -j Prediction -m xgboost -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_six_21_pred.csv -o ./../Test/prediction/logS_testset_six_21_pred.csv -c rdkit_ClogP
time python solubility_model.py -j Prediction -m dnn -dscr rdkit_2d -s clustering -i ./../Test/prediction/logS_testset_six_21_pred.csv -o ./../Test/prediction/logS_testset_six_21_pred.csv -c rdkit_ClogP

