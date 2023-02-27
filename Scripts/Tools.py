import re
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

#from collections import namedtuple
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit import DataStructs

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from torch.utils.data import DataLoader, Dataset



class ESOLCalculator:
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.coef = np.array([0.26121, -0.0066139, -0.74167, 0.0034515, -0.42625])

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mols, names=False):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input list of molecules
        :return: dataframe with four descriptors
        """
        if names:
            assert len(mols) == len(names)
            idx = names
        else:
            idx = list(range(len(mols)))
        
        dscr_names = ["MW", "Logp", "Rotors", "AP"]
        df_dscr = pd.DataFrame(np.nan, index=idx, columns=dscr_names)

        for mol, name in zip(mols, idx):
            if mol:
                df_dscr.loc[name] = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Lipinski.NumRotatableBonds(mol), self.calc_ap(mol)]
        
        return df_dscr


    def calc_esol(self, df):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param df: a dataframe with four descriptors: "MW", "Logp", "Rotors", "AP"
        :return: predicted solubility: dataframe series
        """
        #intercept = 0.26121066137801696
        #coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        df.insert(0, 'Const', 1)
        logS = df.dot(self.coef)
        
        return logS


def Names_generator(DscrNames):
    dscr_names = []
    for Type in DscrNames['Types']:
        for Name in DscrNames[Type]['Names']:
            dscr_names.append(Name)
    return dscr_names

def Values_generator(mol, DscrNames):
    dscr_values = []
    for Type in DscrNames['Types']:
        if Type in ['2D', '3D']:
            for Function in DscrNames['Functions'][Type]:
                dscr_values.append(Function(mol))
        else:
            dscr_values.extend(DscrNames['Functions'][Type](mol))
    return dscr_values

def rdkit_2d_calc(mols, names=False, Autocorr2D=True):

    DescList = Descriptors.descList
    DscrNames = {}
    
    if Autocorr2D == True:
        DscrNames['Types'] = ['2D', 'Autocorr2D']
    else:
        DscrNames['Types'] = ['2D']
        
    DscrNames['Functions'] = {}

    for Type in DscrNames["Types"]:
        DscrNames[Type] = {}
        DscrNames[Type]["Names"] = []
        DscrNames["Functions"][Type] = []
        
    for Type in DscrNames['Types']:
        if Type == '2D':
            for dscr in DescList:
                DscrNames[Type]["Names"].append(dscr[0])
                DscrNames["Functions"][Type].append(dscr[1])
        elif Type == 'Autocorr2D':
            # The current rdkit has 192 Autocorr2D descriptors as of 04_14_2022
            COUNT = 192
            DscrNames[Type]["Names"] = ['Autocorr2D_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcAUTOCORR2D
    
    dscr_names = Names_generator(DscrNames)

    if names:
        assert len(mols) == len(names)
        idx = names
    else:
        idx = list(range(len(mols)))
  
    df_dscr = pd.DataFrame(np.nan, index=idx, columns=dscr_names)
    
    for mol, name in zip(mols, idx):
        if mol:
            df_dscr.loc[name] = Values_generator(mol, DscrNames)
    
    return(df_dscr)
        

def rdkit_3d_calc(mols, names=False):
    Name2Function = {'PMI1' : Descriptors3D.PMI1, 'PMI2' : Descriptors3D.PMI2, 'PMI3' : Descriptors3D.PMI3, 'NPR1' : Descriptors3D.NPR1, 'NPR2' : Descriptors3D.NPR2, 'RadiusOfGyration' : Descriptors3D.RadiusOfGyration, 'InertialShapeFactor' :  Descriptors3D.InertialShapeFactor, 'Eccentricity' : Descriptors3D.Eccentricity, 'Asphericity' : Descriptors3D.Asphericity, 'SpherocityIndex' : Descriptors3D.SpherocityIndex}
    DscrNames = {}
    DscrNames['Types'] = ['Autocorr3D', 'RDF', 'MORSE', 'WHIM', 'GETAWAY', '3D']
    DscrNames['Functions'] = {}
    

    for Type in DscrNames["Types"]:
        DscrNames[Type] = {}
        DscrNames[Type]["Names"] = []
        DscrNames["Functions"][Type] = []

    for Type in DscrNames['Types']:
        if Type == 'Autocorr3D':
            # The current rdkit has 192 Autocorr2D descriptors as of 04_14_2022
            COUNT = 80
            DscrNames[Type]["Names"] = ['Autocorr3D_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcAUTOCORR3D
        elif Type == 'RDF':
            COUNT = 210
            DscrNames[Type]["Names"] = ['RDF_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcRDF
        elif Type == 'MORSE': 
            COUNT = 224
            DscrNames[Type]["Names"] = ['MORSE_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcMORSE
        elif Type == 'WHIM':
            COUNT = 114
            DscrNames[Type]["Names"] = ['WHIM_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcWHIM
            
        elif Type == 'GETAWAY':
            COUNT = 273
            DscrNames[Type]["Names"] = ['GETAWAY_{}'.format(i) for i in range(COUNT)]
            DscrNames["Functions"][Type] = rdMolDescriptors.CalcGETAWAY
            
        else:
            for key, value in Name2Function.items():
                DscrNames[Type]["Names"].append(key)
                DscrNames["Functions"][Type].append(value)  

    dscr_names = Names_generator(DscrNames)
    
    if names:
        assert len(mols) == len(names)
        idx = names
    else:
        idx = list(range(len(mols)))
  
    df_dscr = pd.DataFrame(np.nan, index=idx, columns=dscr_names)
    
    for mol, name in zip(mols, idx):
        if mol:
            df_dscr.loc[name] = Values_generator(mol, DscrNames)
    
    return(df_dscr)
   
def rdkit_2d_3d_calc(mols, names=False):
    df_2d = rdkit_2d_calc(mols, names=names, Autocorr2D=False)
    df_3d = rdkit_3d_calc(mols, names=names)
    df_2d_3d = df_2d.merge(df_3d, left_index=True, right_index=True)

    return df_2d_3d

def rdkit_fps_calc(mols, radius, nbits, names=False):

    fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols]

    if names:
        assert len(mols) == len(names)
        idx = names
    else:
        idx = list(range(len(mols)))
    
    cols = list(range(nbits))
    df_dscr = pd.DataFrame([list(f) for f in fps_list], index=idx, columns=cols)

    return(df_dscr)

def mp_prep(string):
    # extract any chracters(numbers) in between two numbers
    temp = np.nan
    rng = 0
    unit = np.nan
    pat = r'(\d.*\d)'
    string = string.strip()
    string = string.split(';')[0]
    if not (string[0].isdigit() or string[0] == '-'):
        return  (temp, rng, unit)

    else:
        match = re.findall(pat, string)
        if len(match) == 1:
            match = match[0]
            non_match = string.split(match)
        else:
            return  (temp, 0, unit)

        ## process match
        try: 
            if '-' not in match:
                temp = float(match)
            else:
                match_one, match_two = match.split('-')
                match_one_num, match_two_num = float(match_one), float(match_two)
                match_one_dc, match_two_dc = digit_count(match_one_num), digit_count(match_two_num)

                if match_one_dc == match_two_dc:
                    temp = (match_one_num + match_two_num) / 2
                elif abs(match_one_dc - match_two_dc) == 1:
                    if match_one_dc == 3:
                        temp = (match_one_num + 100 * (match_one_num // 100) + match_two_num) / 2
                    else:
                        temp = (match_one_num + 10 * (match_one_num // 10) + match_two_num) / 2
                elif abs(match_one_dc - match_two_dc) == 2:
                    temp = (match_one_num + 10 * (match_one_num // 10) + match_two_num) / 2
                rng = abs(temp - match_one_num)

            if string[0] == '-':
                temp = -temp


            ## process non_match
            if len(non_match) == 1:
                unit_raw = non_match.strip()
            else:
                unit_raw = ' '.join(non_match).strip()
            if 'K' in unit_raw:
                unit = 'K'
            else:
                unit = 'C'
        except: 
            pass
        finally:

            return (temp, rng, unit)


def digit_count(num):
    count = 0
    if type(num) == str:
        num = float(num)
    while num != 0:
        num //= 10
        count += 1
    return count
   

def standardize_temp(T, U):
    abs_k = -273.15
    std_U = 'K'
    if U == 'C':
        T = T - abs_k
    return (T, std_U)


#Getting model/dataset stats for challenged set
def dists_yield(fps, nfps):
    # generator
    for i in range(1, nfps):
        yield [1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]


#def ClusterData(fps, nPts, distThresh, isDistData=False, reordering=False):
def ClusterData(fps, nPts, distThresh, reordering=False):
    """    clusters the data points passed in and returns the list of clusters

        **Arguments**

            - data: a list of items with the input data
                (see discussion of _isDistData_ argument for the exception)

            - nPts: the number of points to be used

            - distThresh: elements within this range of each other are considered
                to be neighbors
            - reodering: if this toggle is set, the number of neighbors is updated
                     for the unassigned molecules after a new cluster is created such
                     that always the molecule with the largest number of unassigned
                     neighbors is selected as the next cluster center.
        **Returns**
            - a tuple of tuples containing information about the clusters:
                 ( (cluster1_elem1, cluster1_elem2, ...),
                     (cluster2_elem1, cluster2_elem2, ...),
                     ...
                 )
                 The first element for each cluster is its centroid.

    """
    nbrLists = [None] * nPts
    for i in range(nPts):
        nbrLists[i] = []

    #dmIdx = 0
    dist_fun = dists_yield(fps, nPts)
    for i in range(1, nPts):
        #print(i)
        dists = next(dist_fun)

        for j in range(i):
            #if not isDistData:
            #    dij = EuclideanDist(data[i], data[j])
            #else:
                #dij = data[dmIdx]
            dij = dists[j]
                #dmIdx += 1
            if dij <= distThresh:
                nbrLists[i].append(j)
                nbrLists[j].append(i)

    # sort by the number of neighbors:
    tLists = [(len(y), x) for x, y in enumerate(nbrLists)]
    tLists.sort(reverse=True)

    res = []
    seen = [0] * nPts
    while tLists:
        _, idx = tLists.pop(0)
        if seen[idx]:
            continue
        tRes = [idx]
        for nbr in nbrLists[idx]:
            if not seen[nbr]:
                tRes.append(nbr)
                seen[nbr] = 1
        # update the number of neighbors:
        # remove all members of the new cluster from the list of
        # neighbors and reorder the tLists
        res.append(tRes)
    return res

def clogP_calc(mols):
    clogp = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        clogp.append(Crippen.MolLogP(mol))
    return clogp

def ClusterFps(fps, method="Auto"):
    #Cluster size is probably smaller if the cut-off is larger. Changing its values between 0.4 and 0.45 makes a lot of difference
    nfps = len(fps)
    
    if method == "Auto":
        if nfps >= 10000:
            method = "TB"
        else:
            method = "Hierarchy"
    
    if method == "TB":
        #from rdkit.ML.Cluster import Butina
        cutoff = 0.56
        print("Butina clustering is selected. Dataset size is:", nfps)

        cs = ClusterData(fps, nfps, cutoff)
        
    elif method == "Hierarchy":
        print("Hierarchical clustering is selected. Dataset size is:", nfps)

        disArray = pdist(fps, 'jaccard')
        #Build model
        Z = hierarchy.linkage(disArray)
        
        #Cut-Tree to get clusters
        #x = hierarchy.cut_tree(Z,height = cutoff)
        average_cluster_size = 8
        cluster_amount = int( nfps / average_cluster_size )     # calculate total amount of clusters by (number of compounds / average cluster size )
        x = hierarchy.cut_tree(Z, n_clusters = cluster_amount )        #use cluster amount as the parameter of this clustering algorithm. 
        
        #change the output format to mimic the output of Butina
        x = [e[0] for e in x]
        cs = [[] for _ in set(x)]

        for i in range(len(x)):
            cs[x[i]].append(i)
    return cs



#function for calculating similarities distributions between two datatsets
def similarity_stat(testSet, trainingSet, bins=10):

    Max = tanimoto_similarity(testSet, trainingSet)
    Max = sum(Max, [])
    hs = np.histogram(Max, bins=bins, range=(0.0, 1.0), density=False)
    Mean = np.mean(Max)
    Std = np.std(Max)

    return hs, Mean, Std

#function for calculating similarities distributions between two datatsets
def tanimoto_similarity(testSet, trainingSet, nn=1):

    Max = []
    #for each compound in the test set, I will calculate the similarity to the nearest (most similar) training set compound
    for i in range(0, len(testSet)):
        sims = DataStructs.BulkTanimotoSimilarity(testSet[i], trainingSet)
        dists = sims
        dists.sort(reverse=True)
        Max.append(dists[0: nn])

    return Max


def ChallengedSplit(data, fp, fraction2train, clusterMethod="Auto"):
    fps = [f for f in data[fp]]
    
    min_select = int(fraction2train * len(data))
    
    cluster_list = ClusterFps(fps, clusterMethod)

    cluster_list.sort(key=len, reverse=True)
    flat_list = sum(cluster_list, [])
    keep_tuple = flat_list[0 : min_select]
    left_tuple = flat_list[min_select : ]
    trn_idx, tst_idx = data.iloc[keep_tuple].index, data.iloc[left_tuple].index
    return trn_idx, tst_idx

def stat(df):
    r2 = df.corr(method='pearson').iloc[0, 1] ** 2
    mae = sum(abs(df.iloc[:, 0] - df.iloc[:, 1])) / len(df)
    rmse = np.sqrt(sum((df.iloc[:, 0] - df.iloc[:, 1]) ** 2) / len(df))
    return(r2, mae, rmse)


def trn_tst_split(df, split, params):
    # std_temp as activity
    std_temp = params['std_temp']
    df_trn = df[df[split] == params['TRN']]
    df_tst = df[df[split] == params['TST']]
    trn_y = df_trn[std_temp]
    tst_y = df_tst[std_temp]
    trn_x = df_trn.drop(labels=[std_temp, split], axis=1)
    tst_x = df_tst.drop(labels=[std_temp, split], axis=1)

    return trn_x, trn_y, tst_x, tst_y


def xgb_prod(df, params, optimized):
    import xgboost as xgb
    # std_temp as activity
    std_temp = params['std_temp']
    trn_y = df[std_temp]
    trn_x = df.drop(labels=[std_temp], axis=1)
    xgboot_model = xgb.XGBRegressor(n_estimators=optimized['n_estimators'], max_depth=optimized['max_depth'], learning_rate=optimized['learning_rate'], colsample_bytree=optimized['colsample_bytree'], n_jobs=params['njobs'], random_state=params['rdn_seed'])
    xgboot_model.fit(trn_x.to_numpy(), trn_y)

    return xgboot_model


def xgb_tune(df, params, params_optim):
    import xgboost as xgb

    split = params['split']

    if split == params['c_split']:
        df.drop(labels=[params['r_split']], axis=1, inplace=True)
    else:
        df.drop(labels=[params['c_split']], axis=1, inplace=True)
    
    trn_x, trn_y, tst_x, tst_y = trn_tst_split(df, split, params)

    n_estimators = params_optim['n_estimators']
    max_depth = params_optim['max_depth']
    learning_rate = params_optim['learning_rate']
    colsample_bytree = params_optim['colsample_bytree']

    stat_names = ['n_estimators', 'max_depth', 'learning_rate', 'colsample_bytree', 'R2', 'MAE', 'RMSE']
    stat_idx = list(range(len(n_estimators) * len(max_depth) * len(learning_rate) * len(colsample_bytree)))

    df_stat = pd.DataFrame(0.0, index=stat_idx, columns=stat_names)
    
    idx = 0
    for tree in n_estimators:
        for depth in max_depth:
                for rate in learning_rate:
                    for colsample in colsample_bytree:
                        
                        #xgboot_model = xgb.XGBRegressor(n_estimators=tree, max_depth=depth, learning_rate=rate, colsample_bytree=colsample, gpu_id=1, n_jobs=njobs, tree_method='gpu_hist', random_state=seed)
                        df_pred = pd.DataFrame(tst_y)
                        xgboot_model = xgb.XGBRegressor(n_estimators=tree, max_depth=depth, learning_rate=rate, colsample_bytree=colsample, n_jobs=params['njobs'], random_state=params['rdn_seed'])
                        xgboot_model.fit(trn_x.to_numpy(), trn_y.to_numpy())
                        df_pred['yhat'] = xgboot_model.predict(tst_x.to_numpy())
                        
                        df_stat.loc[idx, ['n_estimators', 'max_depth', 'learning_rate', 'colsample_bytree']] = [tree, depth, rate, colsample]
                        r2, mae, rmse = stat(df_pred.iloc[:,[0, 1]])
                        print(tree, depth, rate, colsample, r2, mae, rmse)
                        df_stat.loc[idx, ['R2', 'MAE', 'RMSE']] = [r2, mae, rmse]

                        idx += 1

    return df_stat

def rf_tune(df, params, params_optim):
    from sklearn.ensemble import RandomForestRegressor

    split = params['split']

    if split == params['c_split']:
        df.drop(labels=[params['r_split']], axis=1, inplace=True)
    else:
        df.drop(labels=[params['c_split']], axis=1, inplace=True)
    
    trn_x, trn_y, tst_x, tst_y = trn_tst_split(df, split, params)

    n_estimators = params_optim['n_estimators']
    features = params_optim['features']
    split_leaf = params_optim['split_leaf']

    stat_names = ['n_estimators', 'features', 'min_samples_split', 'min_samples_leaf', 'R2', 'MAE', 'RMSE']
    stat_idx = list(range(len(n_estimators) * len(features) * len(split_leaf)))

    df_stat = pd.DataFrame(0.0, index=stat_idx, columns=stat_names)
    
    idx = 0
    for tree in n_estimators:
        for feature in features:
            for sl in split_leaf:
                rf_model = RandomForestRegressor(n_estimators=tree, max_features=feature, min_samples_split=sl[0], min_samples_leaf=sl[1], n_jobs=params['njobs'], random_state=params['rdn_seed'])
                #xgboot_model = xgb.XGBRegressor(n_estimators=tree, max_depth=depth, learning_rate=rate, colsample_bytree=colsample, gpu_id=1, n_jobs=njobs, tree_method='gpu_hist', random_state=seed)
                df_pred = pd.DataFrame(tst_y)
                rf_model.fit(trn_x.to_numpy(), trn_y.to_numpy())
                df_pred['yhat'] = rf_model.predict(tst_x.to_numpy())
                
                df_stat.loc[idx, ['n_estimators', 'features', 'min_samples_split', 'min_samples_leaf']] = [tree, feature, sl[0], sl[1]]
                r2, mae, rmse = stat(df_pred.iloc[:,[0, 1]])
                print(tree, feature, sl[0], sl[1], r2, mae, rmse)
                df_stat.loc[idx, ['R2', 'MAE', 'RMSE']] = [r2, mae, rmse]

                idx += 1

    return df_stat

def rf_prod(df, params, optimized):
    from sklearn.ensemble import RandomForestRegressor
    # std_temp as activity
    std_temp = params['std_temp']
    trn_y = df[std_temp]
    trn_x = df.drop(labels=[std_temp], axis=1)
    rf_model = RandomForestRegressor(n_estimators=optimized['n_estimators'], max_features=optimized['features'], min_samples_split=optimized['min_samples_split'], min_samples_leaf=optimized['min_samples_leaf'], n_jobs=params['njobs'], random_state=params['rdn_seed'])
    rf_model.fit(trn_x.to_numpy(), trn_y)

    return rf_model


class MyDataset(Dataset):

  def __init__(self, X, y):
    self.X = torch.tensor(X.values, dtype=torch.float32)
    self.y = torch.tensor(y.values, dtype=torch.float32)

  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


class TinyModel(nn.Module):

    def __init__(self, inp, layer_1_size, layer_2_size, layer_3_size):
        super(TinyModel, self).__init__()

        self.inp = inp
        self.linear1 = torch.nn.Linear(inp, layer_1_size)
        self.bnm1 = nn.BatchNorm1d(layer_1_size)
        self.linear2 = torch.nn.Linear(layer_1_size, layer_2_size)
        self.bnm2 = nn.BatchNorm1d(layer_2_size)
        self.linear3 = torch.nn.Linear(layer_2_size, layer_3_size)
        self.bnm3 = nn.BatchNorm1d(layer_3_size)
        self.linear4= torch.nn.Linear(layer_3_size, 1)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.LeakyReLU()
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bnm1(x)
        x = self.dropout(self.act2(x))
        x = self.linear2(x)
        x = self.bnm2(x)
        x = self.dropout(self.act2(x))
        x = self.linear3(x)
        x = self.bnm3(x)
        x = self.dropout(self.act2(x))
        x = self.linear4(x)
        return x

class MPDataModule():

    def __init__(self, params):
        super(MPDataModule, self).__init__()
        self.params = params
        #self.fps = '{}/melting_point_fps.job'.format(self.params['data_dirs'])
        self.rdkit_2d = '{}/melting_point_rdkit_2d.job'.format(self.params['data_dirs'])
        self.rdkit_3d = '{}/melting_point_rdkit_3d.job'.format(self.params['data_dirs'])
        self.rdkit_fps = '{}/melting_point_rdkit_fps.job'.format(self.params['data_dirs'])

        
    def df_scaler(self, df):
        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

        cols = [self.params['std_temp'], self.params['r_split'], self.params['c_split']]
        df_act = df[cols]
        df_dscr = df.loc[:, ~df.columns.isin(cols)]

        """
        # **None of below scaler works, rdkit_2d is too complex for those simple scalers**
        if self.params['scaler_name'] == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif self.params['scaler_name'] == 'StandardScaler':
            scaler = StandardScaler()
        elif self.params['scaler_name'] == 'RobustScaler':
            scaler = RobustScaler(quantile_range=(1.0, 99.0))
        elif self.params['scaler_name'] == 'PowerTransformer':
            #PowerTransformer
            #Apply a power transform featurewise to make data more Gaussian-like.
            scaler = PowerTransformer()
        
        if self.params['scaler'] == 'MinMaxScaler':
            scaler = SigmoidScaler2()
        elif self.params['scaler'] == 'StandardScaler':
            scaler = SigmoidScaler()
        """

        scaler = SigmoidScaler()
        scaler._fit(df_dscr)
        joblib.dump(scaler, self.params['scaler_dir'])

        df_dscr = scaler._transform(df_dscr)
        df = df_act.merge(df_dscr, how='left', left_index=True, right_index=True)
        return df
    
    def prepare_dscr(self, df):
        # Caveat: there is no significant difference between modeling using all descriptors and prepared ones. 
        # Users can comment those associated lines related to this function in prepare_data for comparison.
        
        # descriptor preparation
        # remove dscriptors with stdev lower than 0.1 or higher 200
        # remove descriptors that shows low correlation (Peason r2 < 0.001) with the melting point
        # Only keep one descriptors if two or more of them show high collinearity (Peason r2 > 0.95).
        
        std_low_thr = 0.1
        std_high_thr = 200
        #corr_thr = 0.001
        colinear_thr = 0.95

        df_std = df.std(numeric_only=True)
        dscr_remove = list(df_std[(df_std < std_low_thr) | (df_std > std_high_thr)].index)
        df_corr = df.corr(numeric_only=True) ** 2
        df_corr.dropna(axis=0, how='all', inplace=True)
        df_corr.dropna(axis=1, how='all', inplace=True)

        
        #dscr_remove.extend(list(df_corr.std_temp[df_corr.std_temp < corr_thr].index))
        dscr_keep = [col for col in df_corr.columns if col not in dscr_remove]
        #dscr_keep.remove('std_temp')
        
        for dscr in dscr_keep:
            if dscr not in dscr_remove:
                tmp = list(df_corr[dscr][df_corr[dscr] > colinear_thr].index)
                tmp.remove(dscr)
                dscr_remove.extend(tmp)
        dscr_remove = set(dscr_remove)

        if self.params['std_temp'] in dscr_remove:
            dscr_remove.remove(self.params['std_temp'])

        df.drop(dscr_remove, axis=1, inplace=True)

        return df

    def prepare_data(self, prod=False):
        if self.params['dscr'] == 'rdkit_2d':
            df = joblib.load(self.rdkit_2d)
            df = self.prepare_dscr(df)
            df = self.df_scaler(df)

        elif self.params['dscr'] == 'rdkit_3d':
            df = joblib.load(self.rdkit_3d)
            df = self.prepare_dscr(df)
            df = self.df_scaler(df)

        elif self.params['dscr'] == 'rdkit_2d_3d':
            df_2d = joblib.load(self.rdkit_2d)
            df_3d = joblib.load(self.rdkit_3d)
            df_3d.drop(labels=[self.params['std_temp'], self.params['c_split'], self.params['r_split']], axis=1, inplace=True)
            df = df_2d.merge(df_3d, left_index=True, right_index=True)
            del df_2d
            del df_3d
            df = self.prepare_dscr(df)
            df = self.df_scaler(df)

        elif self.params['dscr'] == 'rdkit_fps':
            df = joblib.load(self.rdkit_fps)
        
        
        
        if prod == False:
            # hyperparametr optimization
            split = self.params['split']

            if split == self.params['c_split']:
                df.drop(labels=[self.params['r_split']], axis=1, inplace=True)
            else:
                df.drop(labels=[self.params['c_split']], axis=1, inplace=True)
            
            trn_x, trn_y, tst_x, tst_y = trn_tst_split(df, split, self.params)
            tst_row, tst_col = tst_x.shape
            trn_data = MyDataset(trn_x, trn_y)
            tst_data = MyDataset(tst_x, tst_y)
            
            return trn_data, tst_data, tst_row, tst_col
        else:
            # train a DNN using all data
            trn_y = df[self.params['std_temp']]
            trn_x = df.drop(labels=[self.params['std_temp'], self.params['r_split'], self.params['c_split']], axis=1)
            trn_row, trn_col = trn_x.shape
            trn_data = MyDataset(trn_x, trn_y)
            
            return trn_data, trn_row, trn_col

def _mse(y_hat, y_true):
    return torch.mean(torch.square(y_hat.squeeze(1) - y_true)).item()

def _mae(y_hat, y_true):
    return torch.mean(torch.abs((y_hat.squeeze(1) - y_true))).item()

def _r2(y_hat, y_true):
    y_hat = y_hat.squeeze(1)
    eps = 1e-9
    vx = y_true - torch.mean(y_true)
    vy = y_hat - torch.mean(y_hat)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + eps) * torch.sqrt(torch.sum(vy ** 2) + eps))
    return pcc.item() ** 2


def get_device(gpu):
    # gpu to train the model
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    return device


def dnn_training(model, train_loader, test_loader, optimizer, scheduler, epochs, device):

    mse_loss = nn.MSELoss()
    #epochs = 100
    #epochs = 120
    """
    # Early stopping conditions specific to MP modeling
    a. the test set loss larger than 3000 after 8 epochs
    b. the test set loss larger than the training set loss
    c. the test set loss goes up more than 3 times in a row
    d. 
    """
    last_loss = 100
    patience = 3
    # empirical value only applies to rdkit_2d trainning
    eps = 15
    trigger_times = 0
    
    for epoch in range(epochs):
        model.train()
        model.to(device)
        trn_loss = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(X)
            loss = mse_loss(yhat.squeeze(1), y)
            loss.backward()
            optimizer.step()
            trn_loss.append(loss.item())
        
        #if loss.items()
        model.eval()

        with torch.no_grad():
            """
            tst_loss = []
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                yhat = model(X)
                tst_loss.append(mse_loss(yhat.squeeze(1), y).item())
            tst_loss_mean = np.mean(tst_loss)
            """
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                yhat = model(X)
                tst_loss_mean = mse_loss(yhat.squeeze(1), y).item()

        # schedule the learning rate
        scheduler.step()
        R2 = _r2(yhat, y)
        MAE = _mae(yhat, y)
        RMSE = np.sqrt(tst_loss_mean)

        #print('\tepoch: {}, trn_loss: {:.1f}, tst_loss: {:.1f}, tst_loss2: {:.1f}, R2: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}'.format(epoch, np.mean(trn_loss), tst_loss_mean, tst_loss2, R2, MAE, RMSE))
        print('\tepoch: {}, trn_loss: {:.1f}, tst_loss: {:.1f}, R2: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}'.format(epoch, np.mean(trn_loss), tst_loss_mean, R2, MAE, RMSE))

        # early stopping
        if tst_loss_mean >= last_loss - eps:
            trigger_times += 1
            
            if trigger_times > patience:
                print('\t#Early stopping for the test set loss going up {} times in a row...'.format(patience))
                break
        else:
            trigger_times = 0
        last_loss = tst_loss_mean
        """
        if (epoch >= 10) and (np.mean(trn_loss) <= current_loss):
            #empirical criteria. only specific to 2D descriptor
            print('\t#Early stopping for the training set loss {} lower than the test set loss {}...'.format(trn_loss[-1], current_loss))
            break
        """
        if (epoch >= 8) and (tst_loss_mean > 3200):
            # 3000 is a empirical threshold by considering the all three types of descriptors
            print('\t#Early stopping for the test set loss not converging after {} epochs...'.format(epoch))
            break

    return R2, MAE, RMSE, epoch


def dnn_tune(params, params_optim):

    mp_data = MPDataModule(params)
    trn_data, tst_data, tst_len, tst_col = mp_data.prepare_data(prod=False)
    device = get_device(params['gpu'])

    layer_1_size = params_optim['layer_1_size']
    layer_2_size = params_optim['layer_2_size']
    layer_3_size = params_optim['layer_3_size']
    learning_rate = params_optim['learning_rate']
    batch_size = params_optim['batch_size']

    stat_names = ['layer_1_size', 'layer_2_size', 'layer_3_size', 'learning_rate', 'batch_size', 'R2', 'MAE', 'RMSE', 'epoch']
    stat_idx = list(range(len(layer_1_size) * len(layer_2_size) * len(layer_3_size) * len(learning_rate) * len(batch_size)))

    df_stat = pd.DataFrame(0.0, index=stat_idx, columns=stat_names)
    #INP = params['dscr_num'][params['dscr']]
    INP = tst_col

    count = 0
    for layer_1 in layer_1_size:
        for layer_2 in layer_1_size:
            for layer_3 in layer_1_size:
                for lr in learning_rate:
                    for bs in batch_size:
                        print('Tune:{}, layer_1_size: {}, layer_2_size: {}, layer_3_size: {}, learning_rate: {}, batch_size: {}'.format(count, layer_1, layer_2, layer_3, lr, bs))
                        params_loader = {'batch_size': bs, 'shuffle': True, 'drop_last': True, 'num_workers': 4}
                        train_loader = DataLoader(trn_data, **params_loader)
                        params_loader = {'batch_size': tst_len, 'shuffle': False, 'num_workers': 4}
                        test_loader = DataLoader(tst_data,  **params_loader)
                        model = TinyModel(INP, layer_1, layer_2, layer_3)

                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

                        R2, MAE, RMSE, epoch = dnn_training(model, train_loader, test_loader, optimizer, scheduler, params['epochs'], device)
                        df_stat.loc[count] = [layer_1, layer_2, layer_3, lr, bs, R2, MAE, RMSE, epoch]
                        #yhat, yexpr = dnn_training(model, train_loader, test_loader, optimizer, scheduler, device)
                        count += 1
    return df_stat


def dnn_prod(torch_model, params, params_optim):
    mp_data = MPDataModule(params)
    trn_data, trn_row, trn_col = mp_data.prepare_data(prod=True)
    device = get_device(params['gpu'])

    layer_1_size = params_optim['layer_1_size']
    layer_2_size = params_optim['layer_2_size']
    layer_3_size = params_optim['layer_3_size']
    learning_rate = params_optim['learning_rate']
    batch_size = params_optim['batch_size']
    epochs = params_optim['epoch']
    params_loader = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4}
    train_loader = DataLoader(trn_data, **params_loader)
    #INP = params['dscr_num'][params['dscr']]
    INP = trn_col
    model = TinyModel(INP, layer_1_size, layer_2_size, layer_3_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    mse_loss = nn.MSELoss()
    #epochs = 100
    print('Train a production model with full dataset')
    for epoch in range(epochs + 1):
        model.train()
        model.to(device)
        trn_loss = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(X)
            loss = mse_loss(yhat.squeeze(1), y)
            loss.backward()
            optimizer.step()
            trn_loss.append(loss.item())
 
        # schedule the learning rate
        scheduler.step()
        print('epoch: {}, trn_loss: {}'.format(epoch, np.mean(trn_loss)))
    torch.save(model.state_dict(), torch_model)

def dnn_prediction(params, model_path, df_dscr):
    if params['dscr'] != 'rdkit_fps':
        scaler = joblib.load(params['scaler_dir'])
        dscr = scaler._transform(df_dscr)
    else:
        dscr = df_dscr

    with open(params['network'], 'r') as fid:
            optimized = json.load(fid)

    model = TinyModel(dscr.shape[1], optimized['layer_1_size'], optimized['layer_2_size'], optimized['layer_3_size'])
    # Load the model on cpu for general purpose even though it was trained on GPU
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        yhat = model(torch.tensor(dscr.to_numpy()).float())

    return yhat.squeeze(1).numpy()


class SigmoidScaler3():
    # remove descriptors with stadev less than 0.01
    # sigmod transformation: 1/(1 + np.exp(-c1 * (x - c2)))
    #   x: descriptors to be scaled
    #   c1: 1/stdev, slope determined by standard deviation of x
    #   c2: location determined by median of x
    def __init__(self):
        self.min_thr = 0.01
        self.max_thr = 100
        self.power = 2/3
    
    def _fit(self, df_dscr):
        df_stat = df_dscr.describe()
        self.idx2drop =  list(df_stat.loc['std', (df_stat.loc['std'] < self.min_thr) | (df_stat.loc['std'] > self.max_thr)].index)
        df = df_dscr.drop(self.idx2drop, axis=1)
        df_stat.drop(self.idx2drop, axis=1, inplace=True)
        self.df_std = df_stat.loc['std']
        self.df_median = df_stat.loc['50%']
        #df = 1/(1 + np.exp((-1/np.power(self.df_std, 1/3)) * (df - self.df_median)))
        df = 1/(1 + np.exp((-1/np.power(self.df_std, self.power)) * (df - self.df_median)))
        #self.df_std2 = df.std()
        #self.df_mean = df.mean()

        self.df_min = df.min()
        self.df_max = df.max()

    def _transform(self, df_dscr):
        
        df_dscr = df_dscr.drop(self.idx2drop, axis=1)
        df_dscr = 1/(1 + np.exp((-1/np.power(self.df_std, self.power)) * (df_dscr - self.df_median)))
        #return (df_dscr - self.df_mean) / self.df_std2
        return (df_dscr - self.df_min) / (self.df_max - self.df_min)

class SigmoidScaler():
    # remove descriptors with stadev less than 0.01
    # sigmod transformation: 1/(1 + np.exp(-c1 * (x - c2)))
    #   x: descriptors to be scaled
    #   c1: 1/stdev, slope determined by standard deviation of x
    #   c2: location determined by median of x
    def __init__(self):
        self.power = 2/3

    def _fit(self, df_dscr):
        df_stat = df_dscr.describe()
        self.idx2keep = list(df_dscr.columns)
        self.df_std = df_stat.loc['std']
        self.df_median = df_stat.loc['50%']
        #df = 1/(1 + np.exp((-1/np.power(self.df_std, 1/3)) * (df - self.df_median)))
        df = 1/(1 + np.exp((-1/np.power(self.df_std, self.power)) * (df_dscr - self.df_median)))
        self.df_std2 = df.std(numeric_only=True)
        self.df_mean = df.mean()


    def _transform(self, df_dscr):
        df_dscr = df_dscr[self.idx2keep]
        df_dscr = 1/(1 + np.exp((-1/np.power(self.df_std, self.power)) * (df_dscr - self.df_median)))
        return (df_dscr - self.df_mean) / self.df_std2
        #return (df_dscr - self.df_min) / (self.df_max - self.df_min)
        
class SigmoidScaler2():
    # wrap up the built-in scaler: StandardScaler
    # 
    def __init__(self):
        self.scaler = StandardScaler()

    def _fit(self, df_dscr):
        self.idx2keep =  list(df_dscr.columns)
        self.scaler.fit(df_dscr)

    def _transform(self, df_dscr):
        df_dscr = df_dscr[self.idx2keep]
        idx, cols = df_dscr.index, df_dscr.columns
        self.scaler.transform(df_dscr)
        df_dscr = pd.DataFrame(df_dscr, index=idx, columns=cols)
        return df_dscr
        #return (df_dscr - self.df_min) / (self.df_max - self.df_min)