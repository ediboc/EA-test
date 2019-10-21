# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import itertools
from sklearn.feature_selection import chi2

def calc_clusters(x, cluster_method = 'hierarchy', max_k= 30, linkage_matrix = None, linkage_method='ward', 
                     n_init=20, ini=2):
    
    """
    Calculate vectors of cluster for k from ini to max_k (ex: 2 to 20)
    Return a dataframe with the index as k and the column clusterList 
    
    Parameters
    ----------
    x: vector of points
    
    cluster_method: string {'hierarchy', 'KMeans'}
    
    max_k: maximum number of cluster.
    
    linkage_matrix: Only for 'hierarchy', use a previus linkage_matrix and avoid 
        recalculation
    
    linkage_matrix: string  {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'}
        Only for 'hierarchy', method methods for calculating the distance between 
        the newly formed cluster in calculate linkage_matrix
        
    n_init: int, Only for KMeans
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
        
    ini: int, to start calculate cluster 
    
    """
    #range of k clusters to calculate per method
    nCls = range(ini,max_k+1)
    #data frame to save cluster list per k
    df = pd.DataFrame(columns=['clusterList'],index= nCls)
    
    if cluster_method == 'hierarchy':
        
        if linkage_matrix is None:
            linkage_matrix = linkage(x, metric='euclidean', method='ward')
        distances= linkage_matrix[-max_k:-(ini-1), 2]
        
        for k, max_d in zip(nCls, distances[::-1]):
            df['clusterList'][k] = fcluster(linkage_matrix, max_d, criterion='distance')
    
    elif cluster_method == 'KMeans':
        
        for k in nCls:
            kmean = KMeans(n_clusters = k, n_init = n_init).fit(x)
            df['clusterList'][k] = kmean.labels_
            
    else:
        
        raise ValueError("cluster_method: wrong parameter")
    
    return df


def calc_silhouette(x, df_clusters):
    """
     Calculate silhoutte average
     Return a column with the silhoutte average per cluster
    
    Parameters
    ----------
    x: vector of points
    
    df_clusters: data frame that return the function  "calc_clusters"
    
    """
    
    
    df_clusters['silhouette_avg'] = 0
    
    for k in df_clusters.index:
        df_clusters.loc[k,'silhouette_avg'] = silhouette_score(x, df_clusters.loc[k,'clusterList'])
    
    return df_clusters
    

def plot_silhouette(df_clusters , ini = 2, end = 30, figure_name ="Silhouette plot"):
    """
    Plot silhoutte average with df_clusters
    
    Parameters
    ----------
    df_clusters: data frame that return the function  "calc_silhouette"
    
    ini: int, start plot with cluster k
    
    end: int, end plot with cluster k
    
    figure_name: str, title of plot
     
    """
    ini = max(df_clusters.index.min(),ini)
    end = min(df_clusters.index.max(),end)
    rng = df_clusters.index[(df_clusters.index > ini) & (df_clusters.index < end)]
    plt.plot(rng, df_clusters['silhouette_avg'][rng], 'bx-')
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Sillhoute Avg')
    plt.title(figure_name)
    plt.show()
    
def calc_distorsion(x, df_clusters):
    """
     Calculate distorsion for Elbow method
     Return a column with the distorsion of clusters
    
    Parameters
    ----------
    x: vector of points
    
    df_clusters: data frame that return the function  "calc_clusters"
    
    """
    df_clusters['mean_distorsion'] = 0
    
    for k in df_clusters.index:
        dt = pd.concat([pd.DataFrame(x),pd.DataFrame(df_clusters['clusterList'][k],columns=['cluster'])],axis=1)
        centroid = np.array(dt.groupby('cluster').mean())
        df_clusters.loc[k,'mean_distorsion'] = sum(np.min(cdist(x, centroid,'euclidean'), axis = 1)) / x.shape[0]
        
    return df_clusters
        

def plot_distorsion(df_clusters, ini =1, end =20, figure_name='Distorsion plot'):
    """
    Plot distorsion for Elbow method with df_clusters
    
    Parameters
    ----------
    df_clusters: data frame that return the function  "calc_distorsion"
    
    ini: int, start plot with cluster k
    
    end: int, end plot with cluster k
    
    figure_name: str, title of plot
     
    """
    ini = max(df_clusters.index.min(),ini)
    end = min(df_clusters.index.max(),end)
    rng = df_clusters.index[(df_clusters.index > ini) & (df_clusters.index < end)]
    plt.plot(rng, df_clusters['mean_distorsion'][rng], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Mean Distorsion')
    plt.title(figure_name)        
    plt.show()
        
    
def plot_cluster_matrix(df, clm_clusters = "cluster", clm_categories = "category_id", clm_documents="documents",
                          norm_count="count", norm_colorbar="cluster", aggfunc='count',
                          category_label=None, cluster_label=None,
                          title='Cluster vs Categories - Nº documents', path=None, cmap=plt.cm.BuGn):
    """
    This function prints matrix of nº of documents by cluster*category.
    
    Parameters
    ----------
    df: data frame of documents with categoies and clusters
    
    clm_clusters: string, column with id of cluster.
    
    clm_categories: string, column with id of category.
    
    clm_documents: string, column of document or for count nº of documents.
    
    norm_count: string {'count', 'cluster', 'category'}
        count: show values of documents number
        cluster: show percentage of documents by cluster
        category: show percentage of documents by category
    
    norm_colorbar: string {'count', 'cluster', 'category'}
        count: show color by documents number
        cluster: show color by percentage of documents by cluster
        category: show color by percentage of documents by category
        
    aggfunc: string {'count','sum'}
        count: count rows for clm_documents
        sum: sum values in rows for clm_documents
        
    category_label: string list or pandas  series, xticks
    
    cluster_label:  string list or pandas  series, yticks
        
    cmap: matplotlib colormaps, default: plt.cm.BuGn
    
    savefil: string, name file or/and fold to save plot
    
    """
    #pivot table of documents by cluster and category
    td = pd.pivot_table(df,values=clm_documents, index=clm_clusters, columns=clm_categories, aggfunc=aggfunc).fillna(0)
    
    if norm_count == "cluster":
        td1 = (pd.DataFrame(td.values.T, index=td.columns, columns=td.index)/td.sum(axis=1)).T*100
        tl1 = "Values: normalize nº documents by clusters"
        
    elif norm_count == "category":
        td1 = td/td.sum()*100
        tl1 = "Values: normalize nº documents by categories"   
        
    elif norm_count == "count":
        td1 = td
        tl1 = "Values: nº documents"
    
    else:
        raise ValueError('norm_county: wrong parameter')
        
    td1 = round(td1).astype('int')
        
    
    if norm_colorbar== "cluster":
        td2 = (pd.DataFrame(td.values.T, index=td.columns,columns=td.index)/td.sum(axis=1)).T*100
        tl2 = "Color bar: percentage of tickets by clusters"
        
    elif norm_colorbar == "category":
        td2 = td/td.sum()*100
        tl2 = "Color bar: percentage of tickets by categories" 
        
    elif norm_colorbar == "count":
        td2 = td
        tl2 = "Color bar: nº documents"
    else:
        raise ValueError('norm_colorbar: wrong parameter')
        
    td2 = round(td2).astype('int')
    
    if category_label is None:
        category_label = td.columns
    else:
        category_label = category_label.apply(lambda x: x[:20])
        
    if cluster_label is None:
        cluster_label = td.index
    
    print(tl1,"|",tl2)
    print()
    f = plt.figure(num=None, figsize=(25, 10), dpi=90, facecolor='w', edgecolor='k') # width, height
    plt.imshow(td2, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('percentage of tickets', rotation=270, fontsize=20)
                   
    plt.xticks(np.arange(len(td.columns)), category_label, rotation=90, fontsize=12)
    plt.yticks(np.arange(len(td.index)), cluster_label, fontsize=12)
    thresh = td2.max().max()*.70
    #fmt ='.2f' if normalize else 'f'
    for i, j in itertools.product(td1.index, td1.columns):
        plt.text(j-1, i-td.index.min()+0.5, td1.loc[i, j], fontsize=8,# format(td.loc[i, j], fmt) td.index.min()
                 horizontalalignment="center",
                 color="white" if ((td1.loc[i, j] == 0)| (td2.loc[i, j] > thresh)) else "black") 

#    plt.tight_layout()
    plt.ylabel('Clusters', fontsize=15)
    plt.xlabel('Categories', fontsize=15)
    plt.grid()
    
    if path is not None:
        plt.savefig(path + "/cluster_vs_category.png")
        
    plt.show()


def data_cluster_tokens(data, X_vec, tfidf_model, n_tokens=4, cluster_label = True):
    
    TD = data.groupby(['category_id','category_label','cluster'], as_index=False)['text'].count()
    purity = TD.groupby(['cluster'])['text'].max().sum()/TD.text.sum()
    
    #Calculation of purity
    def Pij(row):
        return -np.log2(row)*row
    #Calculation of entropy
    sumHi = 0
    for cluster in TD.cluster.unique():
        prob = TD.loc[TD.cluster == cluster,'text']/TD.loc[TD.cluster == cluster,'text'].sum()
        Hi = prob.apply(Pij).sum()
        Ni = TD.loc[TD.cluster == cluster,'text'].sum()
        HiNi = Hi*Ni
        sumHi += HiNi
    entropy = sumHi/TD.text.sum()
    
    categories = data['cluster'].unique()
    colnames = []
    for i in range(1,n_tokens+1):  colnames.extend(('Token'+str(i),'weight'+str(i)))
    
    df_main_tokens = pd.DataFrame(columns = colnames,  index = categories)
    
    feature_names = tfidf_model.get_feature_names()
    denselist = X_vec.tolist() 

    for sol in categories:
        
        # compute chi2 for each feature - test how closely each feature is correlated with it's class
        chi2score = chi2(denselist, data['cluster']==sol)[0]
        indices = np.argsort(chi2score)[::-1]
        chi2score_sort = chi2score[indices]
        feature_sol_names = np.array(feature_names)[indices]
   
        for i in range(n_tokens): # we take tha n top tokens          
            
            df_main_tokens.loc[sol,colnames[2*i]] = feature_sol_names[i]
            df_main_tokens.loc[sol,colnames[2*i+1]] = '{:1.2f}'.format(chi2score_sort[i])
        
    df_main_tokens.index.name = 'cluster'
    df_main_tokens.reset_index(inplace=True)
    df_main_tokens = df_main_tokens.sort_values(by='cluster')
    df_main_tokens =  df_main_tokens.merge(data.groupby(['cluster'], as_index=False)['text'].count())
        
    if cluster_label:
    
        df_main_tokens  = pd.DataFrame(data={'cluster_label': df_main_tokens[
                [c for c in df_main_tokens.columns if 'Token' in c]].apply(', '.join,axis=1), 
                'clustey_id': df_main_tokens.cluster, 'n_text':df_main_tokens.text})
    
    print('Purity:',purity)
    print('Entropy:',entropy)
    
    return df_main_tokens
    
 
def Gap_K_means(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters+1)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters+1)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal  

def plotGapStatistic(gapdf,figure_name,k):

    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title(figure_name) #'Gap Values by Cluster Count'  

def plotGapHierarchy(points, figure_name, nCl):
    #points = dataset_to_list_points(dataset)

    # Calculate distances between points or groups of points
    Z = linkage(points, metric='euclidean', method='ward')

    # Obtain the last 10 distances between points
    last = Z[-nCl:, 2]
    num_clustres = np.arange(1, len(last) + 1)

    # Calculate Gap
    gap = np.diff(last, n=2)  # second derivative
    plt.plot(num_clustres[:-2] + 1, gap[::-1], 'ro-', markersize=8, lw=2)
    plt.title(figure_name)
    plt.show()