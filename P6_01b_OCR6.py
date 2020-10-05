import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.savefig('Images/ebouli.png')
    plt.show(block=False)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(20,20))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
            
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, data, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(20,20))
        
            color1=[255/255,140/255,0/255, 1]
            color2=[75/255,0/255,130/255, 1]
            color3=[139/255,69/255,19/255, 1]
            color4=[0/255, 100/255, 0/255, 1]
            color5=[220/255, 20/255, 60/255, 1]

            colormap = np.array([color1, color2, color3, color4, color5])
            
            X_projected = np.hstack((X_projected, np.atleast_2d(data).T))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha, c = colormap[data])
                meanD1 = 0
                meanD2 = 0
                for i in range(int(data.max())+1):
                    meanD1 = X_projected[X_projected[ : , -1] == i][:, d1].mean()
                    meanD2 = X_projected[X_projected[ : , -1] == i][:, d2].mean()
                    plt.scatter(meanD1, meanD2, marker = '^', s = 200, alpha=alpha, c = colormap[i])
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='10', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            
            plt.savefig('Images/factorial_plane_'+str(d1)+'.png')
            plt.show(block=False)
            
def plotbox(df):   
    df_all = df.copy()
    df_all['is_genuine'] = -1
    df_analyse = df.append(df_all, ignore_index=True)
    df_analyse.sort_values(by=['is_genuine'], inplace = True)
    df_analyse.loc[df_analyse['is_genuine'] == -1, 'is_genuine'] = 'All'
    df_analyse.loc[df_analyse['is_genuine'] == 0, 'is_genuine'] = 'Faux Billet'
    df_analyse.loc[df_analyse['is_genuine'] == 1, 'is_genuine'] = 'Vrai Billet'

    for colonne in df_analyse.columns:
        if colonne != 'is_genuine':

            fig, axes = plt.subplots(figsize=(20, 16))

            mu = df_analyse[colonne].mean()
            sigma = df_analyse[colonne].std()

            data_analyse = (df_analyse[colonne] - mu) / sigma

            fig.suptitle('Moyenne de la '+colonne+ ' en fonction du cluster', fontsize= 18)

            ax1 = sns.boxplot(x=df_analyse['is_genuine'], y=data_analyse, showmeans=True)
            ax2 = sns.swarmplot(x=df_analyse['is_genuine'], y=data_analyse, color=".25")
            plt.axvline(0.5)

            plt.xlabel("Etat du Billet")
            plt.ylabel(colonne +" (σ)")

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            i = 0

            for genuine in df_analyse['is_genuine'].unique():

                mu_genuine = df_analyse.loc[df_analyse['is_genuine'] == genuine, colonne].mean()
                sigma_genuine = df_analyse.loc[df_analyse['is_genuine'] == genuine, colonne].std()

                textstr = '\n'.join((
                    r'Population : ',
                    r'$\mu=%.2f$' % (mu_genuine, ),
                    r'$\sigma=%.2f$' % (sigma_genuine, ),
                    r'$n=%.0f$' % (len(df_analyse.loc[df_analyse['is_genuine'] == genuine, colonne]))))

                # place a text box in upper left in axes coords
                # axes.text(0.05 + 0.9 / (len(df_analyse['is_genuine'].unique()) + 1 ) * i, 1.09, textstr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                axes.text(1 / len(df_analyse['is_genuine'].unique()) / 2 + (1 / len(df_analyse['is_genuine'].unique()) * i) - 0.083/2, 1.09, textstr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                i = i + 1

            plt.show()