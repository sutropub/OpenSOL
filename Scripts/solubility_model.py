# This is python 3.9
# Solubility prediction based on the general solubility equation
# rdkit 2D descriptor is used to represent molecules
# xgboost is used to build models based on parameters optimized by clustering and random split

class Model4logS():
    def __init__(self, params):
        self.params = params
        self.fps = '{}/melting_point_fps.job'.format(self.params['data_dirs'])
        self.sdf = '{}/ccdc_mols_melting_point.sdf'.format(self.params['data_dirs'])
        self.rdkit_2d = '{}/melting_point_rdkit_2d.job'.format(self.params['data_dirs'])
        self.rdkit_3d = '{}/melting_point_rdkit_3d.job'.format(self.params['data_dirs'])
        self.rdkit_fps = '{}/melting_point_rdkit_fps.job'.format(self.params['data_dirs'])
        self.mp_data = '{}/melting_point_reference_data.csv'.format(self.params['data_dirs'])
        if params['job'] in ['Modeling', 'Prediction']:
            self.stat_file = '{}/{}_{}_{}_hyperparam_tune_stat.csv'.format(self.params['summary_dirs'], self.params['model'],  self.params['dscr'], self.params['split'])
            if self.params['model'] == 'dnn':
                self.model = '{}/{}_{}_{}_model.pt'.format(self.params['model_dirs'], self.params['model'],  self.params['dscr'], self.params['split'])
                self.params['scaler_dir'] = '{}/{}_{}_scaler.job'.format(self.params['model_dirs'], self.params['model'],  self.params['dscr'])
                self.params['network'] = '{}/{}_{}_{}_network.json'.format(self.params['model_dirs'], self.params['model'],  self.params['dscr'], self.params['split'])
            else:
                self.model = '{}/{}_{}_{}_model.job'.format(self.params['model_dirs'], self.params['model'],  self.params['dscr'], self.params['split'])
        # pytorch models
        
        #self.model_clustering = '{}/mp_xgb_{}_model.job'.format(self.params['model_dirs'], self.params['c_split'])
        #self.model_random = '{}/mp_xgb_{}_model.job'.format(self.params['model_dirs'], self.params['r_split'])
    
    def data_prep(self, file_in):
        # python solubility_model.py -i ../Datasets/_test/test_1000.csv -d ../Datasets/_test -j dataPrep
        smiles = self.params['smiles']
        molecule = self.params['molecule']
        std_temp = self.params['std_temp']
        FP = self.params['FP']
        # fingerprint for clustering and similarity calculation
        nbits = self.params['nbits']
        radius = self.params['radius']
        # fingerprint for modeling
        m_nbits = self.params['m_nbits']
        m_radius = self.params['m_radius']

        frac = self.params['frac']
        c_split = self.params['c_split']
        r_split = self.params['r_split']
        TRN = self.params['TRN']
        TST = self.params['TST']
        rdn_seed = self.params['rdn_seed']
        Autocorr2D = self.params['Autocorr2D']


        if not (os.path.isfile(self.fps) and os.path.isfile(self.rdkit_2d) and os.path.isfile(self.mp_data)): 
            df = pd.read_csv(file_in, index_col=0, header=0)
            print('Number of data points: ', len(df))
            df.dropna(subset=['Melting point', 'SMILES'], axis=0, how='any', inplace=True)
            df[['temp', 'rng', 'unit']] = df.apply(lambda x: list(Tools.mp_prep(x['Melting point'])), axis=1, result_type="expand")
            df.dropna(axis=0, subset=['temp'], inplace=True)
            print('# Data points after removing NaN', len(df))
            df[['std_temp', 'std_unit']] = df.apply(lambda x: list(Tools.standardize_temp(x.temp, x.unit)), axis=1, result_type="expand")

            try:
                PandasTools.AddMoleculeColumnToFrame(df, smiles, molecule)
            except:
                print("Erroneous exception was raised and captured...")
            #remove records with empty molecules
            df = df.loc[df[molecule].notnull()]
            df.drop_duplicates(subset=[smiles, std_temp], keep='first', inplace=True)
            print('# cmpds after dropping duplicates: {}'.format(len(df)))
            # clustering split
            df[FP] = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in df[molecule]]
            # double check to ensure no NA values
            df.dropna(axis=0, how='any', subset=[FP], inplace=True)
            mol_list, ref_code = list(df.molecule), list(df.index)
            df.drop(labels=[molecule], axis=1, inplace=True)

            # Compute rdkit 2D descriptors            
            
            df_2d = Tools.rdkit_2d_calc(mol_list, names=ref_code, Autocorr2D=Autocorr2D)
            
            df_rdkit_fps = Tools.rdkit_fps_calc(mol_list, m_radius, m_nbits, names=ref_code)
            # restore some memory
            del mol_list

            df_2d.dropna(axis=0, how='any', inplace=True)
            df_rdkit_fps.dropna(axis=0, how='any', inplace=True)

            if self.params['calc_3D'] != False:

                df_sdf = PandasTools.LoadSDF(self.sdf)
                mol_3D_list = list(df_sdf[self.params['ROMol']])
                mol_3D_names = list(df_sdf[self.params['ID']])

                del df_sdf
                df_3d = Tools.rdkit_3d_calc(mol_3D_list, names=mol_3D_names)

                del mol_3D_list
                df_3d.dropna(axis=0, how='any', inplace=True)
                comm_idx = list(set(df.index).intersection(set(df_2d.index)).intersection(set(df_3d.index)).intersection(set(df_rdkit_fps.index)))
                df_3d = df_3d.loc[comm_idx]
                
                df_3d[df_3d > 1e5] = 1e5
                df_3d[df_3d < -1e5] = -1e5
            else:
                comm_idx = list(set(df.index).intersection(set(df_2d.index)).intersection(set(df_rdkit_fps.index)))
            print('the number of compounds after processing:', len(comm_idx))
            df = df.loc[comm_idx]
            df_2d = df_2d.loc[comm_idx]
            df_rdkit_fps = df_rdkit_fps.loc[comm_idx]

            # control infinite values that might cause issues
            df_2d[df_2d > 1e5] = 1e5
            df_2d[df_2d < -1e5] = -1e5

            # clustering split
            trn_idx, tst_idx  = Tools.ChallengedSplit(df, FP, frac, clusterMethod="Auto")
            df[c_split] = TRN
            df.loc[tst_idx, c_split] = TST

            # random split
            trn_idx = df.sample(frac=frac, random_state=rdn_seed).index
            df[r_split] = TST
            df.loc[trn_idx, r_split] = TRN


            # save rdkit 2d dataset
            df_2d[std_temp] = df[std_temp]
            df_2d[c_split], df_2d[r_split] = df[c_split], df[r_split]
            joblib.dump(df_2d, self.rdkit_2d)
            
            if self.params['calc_3D'] != False:
                df_3d[std_temp] = df[std_temp]
                df_3d[c_split], df_3d[r_split] = df[c_split], df[r_split]
                joblib.dump(df_3d, self.rdkit_3d)

            df_rdkit_fps[std_temp] = df[std_temp]
            df_rdkit_fps[c_split], df_rdkit_fps[r_split] = df[c_split], df[r_split]
            joblib.dump(df_rdkit_fps, self.rdkit_fps)

            #save fingerprint dataset
            df_fps = df[[smiles, std_temp, FP, c_split, r_split]]
            joblib.dump(df_fps, self.fps)

            # Save reference melting point dataset
            df.drop(labels=[FP], axis=1, inplace=True)
            df.to_csv(self.mp_data, index=True)

    def load_mp_data(self):
        if self.params['dscr'] == 'rdkit_2d':
            return joblib.load(self.rdkit_2d)

        elif self.params['dscr'] == 'rdkit_3d':
            return joblib.load(self.rdkit_3d)
        elif self.params['dscr'] == 'rdkit_2d_3d':
            df_2d = joblib.load(self.rdkit_2d)
            df_3d = joblib.load(self.rdkit_3d)
            df_3d.drop(labels=[self.params['std_temp'], self.params['c_split'], self.params['r_split']], axis=1, inplace=True)
            return df_2d.merge(df_3d, left_index=True, right_index=True)

        elif self.params['dscr'] == 'rdkit_fps':
            return joblib.load(self.rdkit_fps)
        else:
            sys.exit('No descriptor specified')


    def DNN_modeling(self, params_optim):
        #df = self.load_mp_data()

        if not os.path.isfile(self.stat_file):
            df_stat = Tools.dnn_tune(self.params, params_optim)
            df_stat = df_stat.loc[~(df_stat==0.0).all(axis=1)]
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
            df_stat.to_csv(self.stat_file, index=True)
        else:
            df_stat = pd.read_csv(self.stat_file, index_col=0)
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
        
        optimized = {'layer_1_size': int(df_stat.loc[df_stat.index[0], 'layer_1_size']),
                    'layer_2_size': int(df_stat.loc[df_stat.index[0], 'layer_2_size']),
                    'layer_3_size': int(df_stat.loc[df_stat.index[0], 'layer_3_size']),
                    'learning_rate': float(df_stat.loc[df_stat.index[0], 'learning_rate']),
                    "batch_size": int(df_stat.loc[df_stat.index[0], 'batch_size']),
                    "epoch": int(df_stat.loc[df_stat.index[0], 'epoch']),
                    }
        print("Best paramters: ", optimized)

        with open(self.params['network'], 'w') as fid:
            json.dump(optimized, fid)

        ## TODO: finish the produciton code

        if not os.path.isfile(self.model):
            Tools.dnn_prod(self.model, self.params, optimized)
            print('pytorch model saved: {}', self.model)



    def RF_modeling(self, params_optim):
        #
        df = self.load_mp_data()

        if not os.path.isfile(self.stat_file):
            df_stat = Tools.rf_tune(df, self.params, params_optim)
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
            df_stat.to_csv(self.stat_file, index=True)
        else:
            df_stat = pd.read_csv(self.stat_file, index_col=0)
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
        
        try:
            feature = float(df_stat.loc[df_stat.index[0], 'features'])
        except:
            feature = df_stat.loc[df_stat.index[0], 'features']

        optimized = {'n_estimators': int(df_stat.loc[df_stat.index[0], 'n_estimators']),
            'features': feature,
            'min_samples_split': int(df_stat.loc[df_stat.index[0], 'min_samples_split']),
            'min_samples_leaf': int(df_stat.loc[df_stat.index[0], 'min_samples_leaf']),
            }
        
        print("Best paramters: ", optimized)

        if not os.path.isfile(self.model):

            if self.params['r_split'] in df.columns:
                df.drop(labels=[self.params['r_split']], axis=1, inplace=True)
            if self.params['c_split'] in df.columns:
                df.drop(labels=[self.params['c_split']], axis=1, inplace=True)
            
            rf_model = Tools.rf_prod(df, self.params, optimized)

            joblib.dump(rf_model, self.model)

    def xgboost_modeling(self, params_optim):
        # random split best paramters:  {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7, 'njobs': 31, 'rdn_seed': 42}
        # clustering Best paramters:  {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05, 'colsample_bytree': 0.7, 'njobs': 31, 'rdn_seed': 42}
        df = self.load_mp_data()

        if not os.path.isfile(self.stat_file):
            df_stat = Tools.xgb_tune(df, self.params, params_optim)
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
            df_stat.to_csv(self.stat_file, index=True)
        else:
            df_stat = pd.read_csv(self.stat_file, index_col=0)
            df_stat.sort_values(by=['MAE'], axis=0, ascending=True, inplace=True)
        
        # MAE as the objective function here

        optimized = {'n_estimators': int(df_stat.loc[df_stat.index[0], 'n_estimators']),
            'max_depth': int(df_stat.loc[df_stat.index[0], 'max_depth']),
            'learning_rate': df_stat.loc[df_stat.index[0], 'learning_rate'],
            'colsample_bytree': df_stat.loc[df_stat.index[0], 'colsample_bytree'],
            }
        print("Best paramters: ", optimized)

        if not os.path.isfile(self.model):

            if self.params['r_split'] in df.columns:
                df.drop(labels=[self.params['r_split']], axis=1, inplace=True)
            if self.params['c_split'] in df.columns:
                df.drop(labels=[self.params['c_split']], axis=1, inplace=True)
            
            xgb_model = Tools.xgb_prod(df, self.params, optimized)

            joblib.dump(xgb_model, self.model)


    def logS_calc(self, file_in, file_out, clogp):
        # log Sw_solid = 0.5 - 0.01(MP - 298) - log Kow
        # where 298 is room temperature in Kelvin temperature scale
        #Ran, Y., & Yalkowsky, S. H. (2001). Prediction of Drug Solubility by the General Solubility Equation (GSE). Journal of Chemical Information and Computer Sciences, 41(2), 354–357. https://doi.org/10.1021/ci000338c
        # TODO: add sdf reader
        smiles = self.params['smiles']
        molecule = self.params['molecule']
        sept = '\t' if file_in.endswith('.txt') else ','

        if os.path.isfile(file_in):
            df = pd.read_csv(file_in, index_col=None, sep=sept)
            col_names = list(df.columns)
            col_names[0] = smiles
            df.columns = col_names
        else:
            df = pd.DataFrame({smiles: file_in.strip()}, index=['cpd'])
        
        try:
            PandasTools.AddMoleculeColumnToFrame(df, smiles, molecule)
        except:
            print("Erroneous exception was raised and captured...")
        #remove records with empty molecules
        df = df.loc[df[molecule].notnull()]
        df[self.params['clogp']] = Tools.clogP_calc(df[molecule])

        # clustering split

        mols_list = [m for m in df[molecule]]
        # predicting melting point
        #xgboot_model = joblib.load(self.model_random)
        # As shown in preliminary study, 2D is the best.
        # We only compare different algorithms: xgboost, random_forest, dnn
        if self.params['dscr'] == self.params['rdkit_2d']:
            df_dscr = Tools.rdkit_2d_calc(mols_list, names=False, Autocorr2D=self.params['Autocorr2D'])
        elif self.params['dscr'] == self.params['rdkit_3d']:
            df_dscr = Tools.rdkit_3d_calc(mols_list, names=False)
        elif self.params['dscr'] == self.params['rdkit_2d_3d']:
            df_dscr = Tools.rdkit_2d_3d_calc(mols_list, names=False)
        elif self.params['dscr'] == self.params['rdkit_fps']:
            df_dscr = Tools.rdkit_fps_calc(mols_list, self.params['m_radius'], self.params['m_nbits'], names=False)   

        # Processing missing values
        # a. remove molecules with any missing values
        # b. replacing missing values with the average of that particular columns
        '''
        df_dscr.index = df.index
        df_dscr.dropna(axis=0, inplace=True)

        # update df and mols_list for possible failed descriptor calculation
        df = df.loc[df_dscr.index]
        mols_list = [m for m in df[molecule]]
        '''
        df_dscr = df_dscr.fillna(df_dscr.mean(skipna=True))
        #df_dscr = df_dscr.fillna(df_dscr.mode().iloc[0])
        # deal with extreme values
        df_dscr[df_dscr > 1e5] = 1e5
        df_dscr[df_dscr < -1e5] = -1e5

        print('#cpds to predict: {}'.format(len(df_dscr)))

        mp_pred = self.params['model'] + '_' + self.params['MP']

        if self.params['model'] in ['random_forest', 'xgboost']:
            model = joblib.load(self.model)
            df[mp_pred] = model.predict(df_dscr.to_numpy())
        else:
            # dnn model
            df[mp_pred] = Tools.dnn_prediction(self.params, self.model, df_dscr)

        # calculating knn similarityfps_mols = [AllChem.GetMorganFingerprintAsBitVect(m, self.params['radius'], nBits=self.params['nbits']) for m in df[molecule]]
        if (self.params['knn_mean'] not in df.columns) and (self.params['knn_max'] not in df.columns):
            dc = DistCalc(self.params)
            knn_mean, knn_max = dc.knn_similarity(mols_list)
            df[self.params['knn_mean']],  df[self.params['knn_max']] = knn_mean, knn_max

        model_logS = self.params['model'] + '_' + self.params['logS']

        # melting point below room temperature will be set as zero
        df_temp = df[mp_pred] - 298
        df_temp[df_temp < 0] = 0

        try:
            df[model_logS] = 0.5 - 0.01 * df_temp - df[self.params['clogp']]
        except:
            df[model_logS] = 0.5 - 0.01 * df_temp - df[clogp]

        df.drop(labels=[molecule], axis=1, inplace=True)

        df.to_csv(file_out, index=False, sep=sept)

        return df
        

class DistCalc():
    def __init__(self, params):
        self.params = params
        self.fps = '{}/melting_point_fps.job'.format(self.params['data_dirs'])
        self.sim_stat = '{}/mp_trn_tst_sim_stat.csv'.format(self.params['summary_dirs'])
        self.df_fps = joblib.load(self.fps)


    def dist_calc(self):
        splits = [self.params['c_split'], self.params['r_split']]
        bin_space = np.linspace(0, 1, self.params['bins'] + 1)
        hs_head = ['{}-{}'.format(round(i, 2), round(j, 2)) for i, j in zip(bin_space[0 : self.params['bins']], bin_space[1 : self.params['bins'] + 1])]
        col_names = ['Mean', 'Std']
        col_names.extend(hs_head)
        df_sim = pd.DataFrame(0.0, index=splits, columns=col_names)

        for split in splits:
            trn = [f for f in self.df_fps[self.df_fps[split] == self.params['TRN']][self.params['FP']]]
            tst = [f for f in self.df_fps[self.df_fps[split] == self.params['TST']][self.params['FP']]]
            hs, Mean, Std = Tools.similarity_stat(tst, trn, bins=self.params['bins'])
            tmp = [Mean, Std]
            tmp.extend(list(hs[0]))
            df_sim.loc[split, col_names] = tmp
        
        df_sim.to_csv(self.sim_stat, index=True)
    
    def knn_similarity(self, mol):
        if type(mol) != list:
            mol = [mol]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, self.params['radius'], nBits=self.params['nbits']) for m in mol]
        fps_list = [f for f in self.df_fps[self.params['FP']]]
        nn_sim = Tools.tanimoto_similarity(fps, fps_list, nn=self.params['k'])
        
        knn_mean = []
        knn_max = []
        for subset in nn_sim:
            knn_mean.append(np.mean(subset))
            knn_max.append(np.max(subset))
        
        return knn_mean, knn_max

def ESOL_logS(params, file_in, file_out):
    smiles = params['smiles']
    molecule = params['molecule']
    sept = '\t' if file_in.endswith('.txt') else ','

    if os.path.isfile(file_in):
        df = pd.read_csv(file_in, index_col=None, sep=sept)
        col_names = list(df.columns)
        col_names[0] = smiles
        df.columns = col_names
    else:
        df = pd.DataFrame({smiles: file_in.strip()}, index=['cpd'])
    
    try:
        PandasTools.AddMoleculeColumnToFrame(df, smiles, molecule)
    except:
        print("Erroneous exception was raised and captured...")

    df = df.loc[df[molecule].notnull()]
    mols_list = [m for m in df[molecule]]

    esol = Tools.ESOLCalculator()
    df_dscr = esol.calc_esol_descriptors(mols_list)
    df_dscr = df_dscr.fillna(df_dscr.mean(skipna=True))

    model_logS = params['model'] + '_' + params['logS']
    df[model_logS] = esol.calc_esol(df_dscr)

    df.drop(labels=[molecule], axis=1, inplace=True)
    df.to_csv(file_out, index=False, sep=sept)

    return df


def main():

    parser = argparse.ArgumentParser(description='Program for Solubility(logS) Prediction:', epilog="")
    parser.add_argument("--dirs", "-d", help="Directory")
    parser.add_argument("--input", "-i", help="input file")
    parser.add_argument("--output", "-o", help="output file")
    parser.add_argument("--split", "-s", help="split type: clustering or random")
    parser.add_argument("--dscr", "-dscr", help="descriptors: rdkit_2d, rdkit_3d, rdkit_2d_3d, or rdkit_fps")
    parser.add_argument("--gpu", "-g", help="the series number of GPU: 0 or 1")
    parser.add_argument("--njobs", "-n", help="the number of jobs running simultaneously")
    parser.add_argument("--clogP", "-c", help="clogP is available: True or False or a numeric value")
    parser.add_argument("--job", "-j", help="Job to do: dataPred, Modeling, Prediction, distCalc, esol")
    parser.add_argument("--model", "-m", help="machine learning model: random_forest, xgboost, dnn")
    #parser.add_argument("--scaler", "-sc", help="MinMaxScaler, StandardScaler")


    """
    Step 1. compute descriptors
        python solubility_model.py -i ../Datasets/_test/test_1000.csv -d ../Datasets/_test -j dataPrep
    Step 2. build models
        #Build models based on optimized hyperparameters from a clustering split
        python solubility_model.py -i ../Datasets/_test/melting_point_rdkit_2d_n_400.job -o ../Models/_test/xgb_model_clustering_test.job -s clustering -d ../Models/_test -n 30 -j xgboost
        #Build models based on optimized hyperparameters from a random split
        python solubility_model.py -i ../Datasets/_test/melting_point_rdkit_2d_n_400.job -o ../Models/_test/xgb_model_random_test.job -s random -d ../Models/_test -n 30 -j xgboost
    Step 3. compute distance

    Step 4. make prediction
    """
    params = {
        'data_dirs': '../Datasets',
        'model_dirs': '../Models',
        'summary_dirs': '../Summary',
        'molecule': 'molecule',
        'smiles': 'SMILES',
        'FP': 'FP',
        'TRN': 'TRN',
        'TST': 'TST',
        'rdkit_2d': 'rdkit_2d',
        'rdkit_3d': 'rdkit_3d',
        'rdkit_2d_3d': 'rdkit_2d_3d',
        'rdkit_fps': 'rdkit_fps',
        'calc_3D': False, # set to False 
        'c_split': 'clustering',
        'r_split': 'random',
        'split': 'random',
        'clogp': 'rdkit_ClogP',
        'std_temp': 'std_temp',
        'Autocorr2D': False,
        'frac': 0.75,
        'rdn_seed': 42,
        # radius and nbits for similarity calculation
        'radius': 2,
        'nbits': 1024,
        # radius and nbits for modeling
        'm_radius': 3,
        'm_nbits': 2048,
        'dscr_num': {'rdkit_2d': 208, 'rdkit_3d': 911, 'rdkit_2d_3d': 1119, 'rdkit_fps': 2048},
    }

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    params['job'] = args.job

    if args.job == 'dataPrep':
        params['ROMol'] = 'ROMol'
        params['ID'] = 'ID'
        file_in = args.input
        dp = Model4logS(params)
        dp.data_prep(file_in)

        sys.exit('Data preparation done...')

    if args.job == 'Modeling':
        params['model'] = args.model
        params['dscr'] = args.dscr
        params['split'] = args.split

        if args.njobs:
            params['njobs'] = int(args.njobs)

        if args.model == 'random_forest':
            params_optim = {
                'n_estimators': [500],
                'features': ['sqrt', 0.33],
                'split_leaf': [[2, 1], [5, 3], [7, 4]],
                }
            
            rf = Model4logS(params)
            rf.RF_modeling(params_optim)
        
        if args.model == 'xgboost':
            params_optim = {
                'n_estimators': [500],
                'max_depth': [5, 8, 10, 12, 15],
                'learning_rate': [0.03, 0.05, 0.07, 0.1],
                'colsample_bytree': [0.5, 0.7, 0.9],
                }
            xm = Model4logS(params)
            xm.xgboost_modeling(params_optim)

            sys.exit('Model building done...')

        if args.model == 'dnn':
            print('Building DNN models')

            params['epochs'] = 25
            #params['scaler_name'] = args.scaler
            #params['scaler_name'] = 'RobustScaler'
            #params['scaler_name'] = 'MinMaxScaler' #StandardScaler, RobustScaler
            #params['scaler'] = args.scaler

            if args.gpu:
                # GPU series number 0 or 1 on af.cellfree
                if args.gpu in ['0', '1']:
                    params['gpu'] = args.gpu
                else:
                    params['gpu'] = '0'

            #params['dropout'] = 0.25
            
            if params['dscr'] == 'rdkit_2d':
                params_optim = {
                    'layer_1_size': [128, 256],
                    'layer_2_size': [128, 256, 512],
                    'layer_3_size': [128, 256],
                    'learning_rate': [1e-4, 5e-4, 1e-3],
                    "batch_size": [16, 32, 64],
                    }
            elif params['dscr'] in ['rdkit_3d', 'rdkit_2d_3d']:
                params_optim = {
                    'layer_1_size': [512, 1024],
                    'layer_2_size': [512, 1024, 2048],
                    'layer_3_size': [512, 1024],
                    'learning_rate': [1e-4, 5e-4, 1e-3],
                    "batch_size": [16, 32, 64],
                    }
            elif params['dscr'] == 'rdkit_fps':
                params_optim = {
                    'layer_1_size': [1024, 2048],
                    'layer_2_size': [1024, 2048, 4096],
                    'layer_3_size': [1024, 2048],
                    'learning_rate': [1e-4, 5e-4, 1e-3],
                    "batch_size": [16, 32, 64],
                    }
            else:
                sys.exit('No descriptor specified')
            
            dnn = Model4logS(params)
            dnn.DNN_modeling(params_optim)


    if args.job == 'distCalc':
        params['bins'] = 10

        calc = DistCalc(params)
        calc.dist_calc()
        sys.exit('Similarity calculation done...')

    if args.job == 'Prediction':
        # a plain text file with smiles or one smile string
        """
        file_in: smiles in the first column is enssential, it can be than one column.
        smi,clogP
        c1ccccc1,3.0
        CCCNCC(=O)O,2.0

        """
        file_in = args.input
        file_out = args.output

        params['model'] = args.model
        params['logS'] = 'logS'

        if params['model'] != 'esol':
            clogp = args.clogP
            params['dscr'] = args.dscr
            params['split'] = args.split
            params['MP'] = 'MP'
            params['k'] = 5
            params['knn_mean'] = 'knn_mean'
            params['knn_max'] = 'knn_max'
            cs = Model4logS(params)
            cs.logS_calc(file_in, file_out, clogp)
        else:
            ESOL_logS(params, file_in, file_out)


if __name__ == "__main__":
    import os
    import sys
    import argparse
    import json
    import joblib

    import numpy as np
    import pandas as pd

    import Tools
    
    from rdkit.Chem import AllChem
    from rdkit.Chem import PandasTools
    main()
