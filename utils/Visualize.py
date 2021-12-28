# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:27:05 2021

@author: mahdi
"""

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from utils.Global_Variable import Electrode
import numpy as np
from scipy import interpolate

def Visualize(Brain_Map_Instance, Num_Points=200):
    
    #Set the size
    Length=Brain_Map_Instance.shape[0]
    #Sets the points of outter head
    O=30
    # gets Brain-Map single trial and single time
    Elev=[]
    points=[]
    newshapex=[]
    newshapey=[]
    #Loads Electrode Elements
    Electrode_Pos=Electrode.Get_Elec_Postions()
    for pos in range(len(Electrode_Pos)):
        if Electrode_Pos[pos] !=0:
            points.append([ int(np.floor(pos/Length)),int(pos%Length)])
            Elev.append(Brain_Map_Instance[int(np.floor(pos/Length))][int(pos%Length)])
            newshapex.append(int(np.floor(pos/Length)))
            newshapey.append(int(pos%Length))
    #Outer Head Points
    Outline=np.linspace(0,2*np.pi,O)
    Outerx=[]
    Outery=[]
    Outer=[]
    for h in range(len(Outline)):
        Outerx.append(3+3*np.cos(Outline[h]))
        Outery.append(3+3*np.sin(Outline[h]))
        Outer.append([3+3*np.cos(Outline[h]),3+3*np.sin(Outline[h])])
    [x,y]=np.meshgrid(np.linspace(0,Length,Num_Points),np.linspace(0,Length,Num_Points))
    #ExtraPolate and add points
    f = interpolate.interp2d(newshapex, newshapey, Elev, kind='cubic')
    New=[]
    for i in range(O):
        New.append(f(Outerx[i],Outery[i]).tolist()[0])        
    Elev=Elev + New
    points+=Outer
    #Grid and plot the inverse
    z = griddata(points, Elev, (x, y), method='linear')
    plt.contour(y, x,z,colors='black')
    x = np.matrix.flatten(x); #Gridded longitude
    y = np.matrix.flatten(y); #Gridded latitude
    z = np.matrix.flatten(z); #Gridded elevation
    #Invert Axis and set axis
    plt.scatter(y,x,2,z)
    plt.axis([0, 6, 0, 6])
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.axis('off')
    plt.colorbar(label='Power')
    plt.show()
    
if __name__=='__main__':
    pass