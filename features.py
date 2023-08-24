"""Point Cloud Features
Requirements:
  - pandas
  - numpy
  - skimage
  - sklearn
  - math
  - seaborn
  - scipy

@author: chaconnb

Python:
  - 3.8.5
 """

import pandas as pd
import numpy as np
from skimage import color
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import math
import seaborn as sns
from scipy.stats import skew, kurtosis


def subsamplecloud(point_cloud_XYZ,method,numps_subfrac,*args): 

    '''
    Subsamples the point cloud.

            Parameters:
                    point_cloud_XYZ(pandas data frame): X,Y,Z coordinates of point cloud
                    method(int): 0 = balanced subsample for core points within labeled points (*args =  labels)
                               # 1 = randomly sample the entire point cloud (no labels)
                               # 2 = randomly sample the entire point cloud (*args =  labels)
                    numps_subfrac(int) : percentage of points subsampled if subsampleflag == 1 OR percentage of points subsampled PER CLASS if subsampleflag == 0 
                    *args(pandas data frame) : column of labels if method = 0 or 2. 

            Returns:
                    pts_core (pandas data frame): points subsampled used for the feature generation
                    numicore (inte): number of pts_core
    '''

    if method == 1:
        pts_core= point_cloud_XYZ.sample(frac = numps_subfrac,random_state=1) #random state is used for error finding purposes
        numicore=pts_core.shape[0]
    
    elif method == 2:
        for i in args:
             label = i
        frames = [point_cloud_XYZ,label]  
        point_cloud_XYZ = pd.concat(frames,axis=1)
        pts_core= point_cloud_XYZ.sample(frac = numps_subfrac,random_state=1)  #random state is used for error finding purposes
        numicore=pts_core.shape[0]
        
    else:
        for i in args:
             label = i.squeeze()
        
        pointcXYZ = point_cloud_XYZ.copy() #generate a copy so the original data frame is not affected 
        label = label.rename('label')
        label = label[label != 2]
        thruth = label.unique()
        freq = label.value_counts().to_frame().sort_values(by=['label']) 
        min_freq = freq['label'].iloc[0]

        N = int(math.ceil(min_freq*numps_subfrac))
        pts_core=[]

        for i in thruth:
            pointcXYZ.loc[:,'labels'] = label
            dfi =  pointcXYZ.loc[pointcXYZ['labels']==i]
            pts_core.append(dfi.sample(N,random_state=1))

        pts_core = pd.concat(pts_core)
        numicore=pts_core.shape[0]
     
    return(pts_core,numicore)


def gsc_descriptors(point_cloud_XYZ,col,typecolor,method,numps_subfrac,radius,normal,*args):

    '''
    Subsamples the point cloud.

            Parameters:
                    point_cloud_XYZ(pandas data frame): X,Y,Z coordinates of point cloud
                    col(pandas data frame): R,G,B OR L,a,b colors 
                    typecolor : 1 = RGB
                                0 = Lab
                    method(int): 0 = balanced subsample for core points within labeled points (*args =  labels)
                               # 1 = randomly sample the entire point cloud (no labels)
                               # 2 = randomly sample the entire point cloud (*args =  labels)
                    numps_subfrac(int): percentage of points subsampled if subsampleflag == 1 OR percentage of points subsampled PER CLASS if subsampleflag == 0
                    radius(int): radius for feature calculation
                    normal(pandas data frame): Z component Normal vector 
                    *args(pandas data frame) : column of labels if method = 0 or 2. 

            Returns:
                   output(pandas data frame): data frame with features.

    '''
    
    pca = PCA(n_components = 3)
   
    
    #Core points 
    if method == 1:
        pts_core,numicore = subsamplecloud(point_cloud_XYZ,method,numps_subfrac)

    
    elif method == 2:
        for i in args:
             label = i
        pts_core,numicore = subsamplecloud(point_cloud_XYZ,method,numps_subfrac,label)
        labels = pts_core.iloc[:,3].to_frame() #data frame
        pts_core = pts_core.iloc[:,0:3]
        
    else:
        for i in args:
             label = i
        pts_core,numicore = subsamplecloud(point_cloud_XYZ,method,numps_subfrac,label)
        labels = pts_core.iloc[:,3].to_frame() #data frame
        pts_core = pts_core.iloc[:,0:3]
#         point_cloud_XYZ = point_cloud_XYZ.iloc[:,0:3] #Este arreglo medio chambon lo hice para remediar el problema en subsample qu

     #Color
    if typecolor == 0:
        L = col.iloc[:,0].to_numpy()
        a = col.iloc[:,1].to_numpy()
        b = col.iloc[:,2].to_numpy()
        Lab = col.to_numpy()
                                
    else:
        R = col.iloc[:,0].to_numpy()
        G = col.iloc[:,1].to_numpy()
        B = col.iloc[:,2].to_numpy()                 
                
        RGB = np.transpose(np.array([R,G,B]))
        Lab = color.rgb2lab(RGB)
                
        L = [row[0] for row in Lab]
        a = [row[1] for row in Lab]
        b = [row[2] for row in Lab]
   
    
    #find the verticality 
    ver = (1 - normal).abs()
    ver = ver.to_numpy() #data frame to numpy
    
    #search index of the core point to search the LAB value and calculte the relative color features
    #reset index as another column
    pts_core.reset_index(level=0, inplace = True)
    

   #place index in the last column of the data frame
    cols = pts_core.columns.tolist()
    cols = cols[1:] + cols[:-3]
    pts_core = pts_core[cols]
    
    
    pc_array = point_cloud_XYZ.to_numpy()
    
    pts_core_array = pts_core.to_numpy()
    color_array = col.to_numpy()
    
    
    tree = BallTree(pc_array, leaf_size= 65)
    
    
    #Geometric Features
    eigenvalues = np.zeros((numicore,3))
    omni = np.zeros((numicore, 1))
    eigent = np.zeros((numicore,1))
    aniso = np.zeros((numicore, 1))
    planar = np.zeros((numicore, 1))
    linear = np.zeros((numicore, 1))
    curv = np.zeros((numicore, 1))
    scatt = np.zeros((numicore, 1))
    
    #Slope Features
    vert_mean = np.zeros((numicore,1))
    vert_std = np.zeros((numicore, 1))
    vert_skew = np.zeros((numicore,1))
    vert_kurt = np.zeros((numicore, 1))
    
    #Color LAB
    LAB_i = np.zeros((numicore,3))
    color_std = np.zeros((numicore,3))
    color_mean = np.zeros((numicore,3))

    #relative_color_features
    L_mean_point = np.zeros((numicore,1))
    A_mean_point = np.zeros((numicore,1))
    B_mean_point = np.zeros((numicore,1))

    L_point_min = np.zeros((numicore,1))
    A_point_min = np.zeros((numicore,1))
    B_point_min = np.zeros((numicore,1))

    L_point_max = np.zeros((numicore,1))
    A_point_max = np.zeros((numicore,1))
    B_point_max = np.zeros((numicore,1))
    
    #g8 features Walton
    L_sum_num = np.zeros((numicore,1)) #1
    A_sum_num = np.zeros((numicore,1)) #1 
    B_sum_num = np.zeros((numicore,1)) #1
    
    Lratio_sum_vol = np.zeros((numicore,1)) #2
    Aratio_sum_vol = np.zeros((numicore,1)) #2
    Bratio_sum_vol = np.zeros((numicore,1)) #2
    
    L_eig_scal = np.zeros((numicore,2)) #3
    A_eig_scal = np.zeros((numicore,2)) #3
    B_eig_scal = np.zeros((numicore,2)) #3
    
    L_xy_eig = np.zeros((numicore,4)) #4
    A_xy_eig = np.zeros((numicore,4)) #4
    B_xy_eig = np.zeros((numicore,4)) #4
    
    for i in range(numicore):
        core_point = pts_core_array[:,0:3][i]
        ind = tree.query_radius([core_point], r=radius) 
        
    
        if len(ind[0])>2:
            
            neigh_coords = [[pc_array[ind[0][j]][0],pc_array[ind[0][j]][1], pc_array[ind[0][j]][2]] for j in range(len(ind[0]))]
                                
            
            pca.fit(neigh_coords)
        
            #Calculate eigenvalues
            L1, L2, L3 = pca.explained_variance_ratio_
            eigenvalues[i,0:3] = pca.explained_variance_ratio_
            
            
            #Calculate Geometric Features
            omni[i] = (L1*L2*L3)**(1/3)
            eigent[i] = -(L1*np.log(L1)+L2*np.log(L2)+L3*np.log(L3))
            aniso[i] = (L1-L3)/L1
            planar[i] = (L2-L3)/L1
            linear[i] = (L1-L2)/L1
            curv[i] = L3
            scatt[i] = L3/L1
            
            #Calculate Verticality Features
            vert = [ver[i] for i in ind[0]]
            vert_mean[i] = np.mean(vert)
            vert_std[i] = np.std(vert)
            vert_skew[i] = skew(vert)
            vert_kurt[i] = kurtosis(vert)
            
            
         #Calculate Color Features
            general_index = pts_core_array[i][3].astype(np.int64)
            LAB_i[i,:] =Lab[general_index]
        
            neigh_color_L = [L[i] for i in ind[0]]
            neigh_color_A = [a[i] for i in ind[0]]
            neigh_color_B = [b[i] for i in ind[0]]
            
            neigh_color = [Lab[i] for i in ind[0]]
            
            
           #stdev color
            color_std[i,0:3] = [np.std(neigh_color_L), np.std(neigh_color_A), np.std(neigh_color_B)]


          #mean color
            color_mean[i,0:3] = [np.mean(neigh_color_L), np.mean(neigh_color_A), np.mean(neigh_color_B)]
            
            
          #relative color features(Walton,2016)
            
           #L #Mean
            
            L_mean_point[i] = np.mean(neigh_color_L) - LAB_i[i,0]
            #point min
            L_point_min[i] = LAB_i[i,0] - min(neigh_color_L)
            #point max
            L_point_max[i]= max(neigh_color_L) - LAB_i[i,0]
            
            #A #Mean
            A_mean_point[i] = np.mean(neigh_color_A) - LAB_i[i,1]
            #point min
            A_point_min[i]= LAB_i[i,1] - min(neigh_color_A)
            #point max
            A_point_max[i]= max(neigh_color_A)- LAB_i[i,1]
            
            #B #Mean
            B_mean_point[i] = np.mean(neigh_color_B) - LAB_i[i,2]
            #point min
            B_point_min[i]= LAB_i[i,2] - min(neigh_color_B)
            #point max
            B_point_max[i]= max(neigh_color_B) - LAB_i[i,2]
        
        
            
            #g8 features Walton
            
            len_ind = len(ind[0])
            conv = 100*100*100
            volu_neigh =conv* 4/3*math.pi*radius**3 #calculate the volume of the neighboorhood per cm**3
            
            #1 The sum of the neighborhood intensity values divided by the neighborhood volume (intensity weighted density)
            
            L_sum_num[i] = np.sum(neigh_color_L)/volu_neigh
            A_sum_num[i] = np.sum(neigh_color_A)/volu_neigh
            B_sum_num[i] = np.sum(neigh_color_B)/volu_neigh
            
            #2 The ratio of the intensity weighted density to the point density (defined as the # of points in the neighborhood divided 
            #by the neighborhood volume)
            
            Lratio_sum_vol[i] = L_sum_num[i]*volu_neigh/len_ind
            Aratio_sum_vol[i] = A_sum_num[i]*volu_neigh/len_ind
            Bratio_sum_vol[i] = B_sum_num[i]*volu_neigh/len_ind
            
            
            #3  The two largest normalized eigenvalues of the weighted covariance matrix(computed by scaling each point's (x,y,z) vector from
            # the geometric center of the neighbohood by its intensity value). 
            #4 The x and y (horizontal) components of the normalized eigenvectors corresponding to the two largest eigenvalues.
            
            deltap = neigh_coords-core_point
            
            deltap_L = deltap * (np.vstack((neigh_color_L,neigh_color_L,neigh_color_L)).T)
            deltap_A = deltap * (np.vstack((neigh_color_A,neigh_color_A,neigh_color_A)).T)
            deltap_B = deltap * (np.vstack((neigh_color_B,neigh_color_B,neigh_color_B)).T)
            
            #find the two largest normalized eigenvalues of deltap_L, deltap_A and deltap_B
            
            pca_deltap = PCA(n_components = 2)
            
            #deltap_L
            
            pca_deltap.fit(deltap_L)
            #3
            L_eig_scal[i,0:2] = pca_deltap.explained_variance_ratio_ #normalized eigenvalue
            #4
            eigL = pca_deltap.components_ #eigenvectors
            L_xy_eig [i,:] = eigL[:,0:2].flatten() #only x and y components ?????(I am not sure this are the y and x components)
            
            
            #deltap_A
            
            pca_deltap.fit(deltap_A)
            #3
            A_eig_scal[i,0:2] = pca_deltap.explained_variance_ratio_ #normalized eigenvalue
            #4
            eigL = pca_deltap.components_ #eigenvectors
            A_xy_eig [i,:] = eigL[:,0:2].flatten() #only x and y components ?????(I am not sure this are the y and x components)
            
            #deltap_B
            
            pca_deltap.fit(deltap_B)
            #3
            B_eig_scal[i,0:2] = pca_deltap.explained_variance_ratio_ #normalized eigenvalue
            #4
            eigL = pca_deltap.components_ #eigenvectors
            B_xy_eig [i,:] = eigL[:,0:2].flatten() #only x and y components ?????(I am not sure this are the y and x components)
                   

    output1 = pd.DataFrame(zip(pts_core_array[:,0],pts_core_array[:,1],pts_core_array[:,2],LAB_i[:,0],LAB_i[:,1],LAB_i[:,2],eigenvalues[:,0],eigenvalues[:,1],eigenvalues[:,2],
                               omni[:,0], eigent[:,0],aniso[:,0],planar[:,0],linear[:,0],curv[:,0],scatt[:,0],vert_mean[:,0],vert_std[:,0],
                               vert_skew[:,0],vert_kurt[:,0],color_std[:,0],color_std[:,1],color_std[:,2],color_mean[:,0],color_mean[:,1],
                               color_mean[:,2],L_mean_point[:,0],L_point_min[:,0],L_point_max[:,0],A_mean_point[:,0],A_point_min[:,0],
                               A_point_max[:,0],B_mean_point[:,0],B_point_min[:,0],B_point_max[:,0],L_sum_num[:,0],A_sum_num[:,0],
                               B_sum_num[:,0],Lratio_sum_vol[:,0],Aratio_sum_vol[:,0],Bratio_sum_vol[:,0],L_eig_scal[:,0],L_eig_scal[:,1]
                               ,A_eig_scal[:,0],A_eig_scal[:,1],
                               B_eig_scal[:,0],B_eig_scal[:,1],L_xy_eig[:,0],L_xy_eig[:,1],L_xy_eig[:,2],L_xy_eig[:,3],A_xy_eig[:,0],
                               A_xy_eig[:,1],A_xy_eig[:,2],A_xy_eig[:,3],B_xy_eig[:,0],B_xy_eig[:,1],B_xy_eig[:,2],B_xy_eig[:,3]),

                           columns = ['X','Y','Z','L','a','b','L1','L2', 'L3','Omnivariance','Eigenentropy','Anisotropy','Planarity','Linearity',
                                      'Surface variation','Scatter','verticality_mean','verticality_std','verticality_skew','verticality_kurtosis','Color_L std',
                                      'Color_a std','Color_b std','Color_L mean','Color_a mean','Color_b mean','L mean point', 'L min point',
                                       'L max point','A mean point', 'A min point','A max point','B mean point', 'B min point','B max point',
                                        'L_sum_num','A_sum_num','B_sum_num','L_ratio_sum_vol','A_ratio_sum_vol','B_ratio_sum_vol','largenormeig 1 L','largenormeig 2 L','largenormeig 1 A',
                                      'largenormeig 2 A','largenormeig 1 B','largenormeig 2 B','L1_x_eig','L1_y_eig','L2_x_eig','L2_y_eig',
                                      'A1_x_eig','A1_y_eig','A2_x_eig','A2_y_eig','B1_x_eig','B1_y_eig','B2_x_eig','B2_y_eig']) 

    if method == 1:
        output = output1
    else:
        labels = labels.reset_index(drop=True)
        output = pd.concat([output1,labels],axis=1)
    
    output.to_csv('{}.txt'.format(radius))  
    
    return(output)