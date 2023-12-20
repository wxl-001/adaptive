import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import *
from scipy.optimize import curve_fit
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import r2_score

class IAdaptNetwork1:#The initial network is generated randomly.
    def __init__(self,k1,k2,k3,phi,L,N,N1):
        #k1=k+-,k2=k++,k3=k--,phi=the frequency of Voter model,L=#(links),N=#(Nodes),N1=#(+ opinion)
        self.k1=k1
        self.k2=k2
        self.k3=k3
        #k1=kd,k2=k3=ks
        self.L=L
        self.phi=phi
        self.N=N
        assert self.L<self.N*(self.N-1)/2,"Too many links!!!"
        self.k=2*self.L/self.N#k=average degree
        assert N1<N,"N1<N!!!!!"
        self.N1=N1
        self.x=self.N1/self.N#x=the frequency of +opinion
        self.opinion=self.generate_opinion()
        self.link=self.generate_link()
        self.adjacency=self.generate_adjacency()
        self.u=self.generate_u()
        self.u0=self.u
        #print("Initial u:",self.u)
        #print("u*=",self.fu())
        self.v=self.generate_v()
        self.w=self.generate_w()
        self.stat=[]
        self.ini_opinion=self.opinion.copy()
        self.ini_adjacency=self.adjacency.copy()
        
    def generate_opinion(self):
        opinion=np.array([-1]*self.N,dtype=np.int64)
        opinion[random.sample(range(self.N),self.N1)]=1
        return opinion
    
    def generate_link(self):
        link=[]
        for i in range(self.N):
            for j in range(i):#j<i
                link.append((i,j))
        #print("link:",link)
        #a=random.sample(link,self.L)
        #print("a",a)
        return random.sample(link,self.L)
        
    def generate_adjacency(self):
        adjacency=np.zeros((self.N,self.N),dtype=np.int64)
        for point in self.link:
            i,j=point
            adjacency[(i,j)]=1
            adjacency[(j,i)]=1
        return adjacency
        
    def generate_AA(self):
        AA=0
        for point in self.link:
            i,j=point
            #print(self.adjacency[i][j])
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==2:
                AA+=1
        return AA
    
    def generate_BB(self):
        BB=0
        for point in self.link:
            i,j=point
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==-2:
                BB+=1
        return BB
    
    def generate__AB(self):
        AB=0
        for point in self.link:
            i,j=point
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==0:
                AB+=1
        return AB
    
    def generate_A(self):
        return len(self.opinion[self.opinion>0])
    
    def generate_B(self):
        return len(self.opinion[self.opinion<0])
    
    def generate_u(self):
        return (self.generate_AA()+self.generate_BB())/self.L
    
    def generate_v(self):
        return (self.generate_AA()-self.generate_BB())/self.L
    
    def generate_w(self):
        return (self.generate_A()-self.generate_B())/self.N
    
    def rdom(self,threshold):#0<threshold<1
        return random.random()<threshold
    
    def choose_node1(self,i):#No adjacent nodes
        total=[]
        for j in range(self.N):
            if self.adjacency[(i,j)]==0 and j!=i:
                total.append(j)
        assert len(total)>0,"Impossible!!!!"
        return random.choice(total)
    
    def choose_node2(self,i):#Adjacent nodes
        total=[]
        for j in range(self.N):
            if self.adjacency[(i,j)]==1:
                total.append(j)
        if len(total)==0:
            return -1
        return random.choice(total)
    
    def linking(self):
        i,j=random.choice(self.link)
        if self.opinion[i]+self.opinion[j]==0:
            k=self.k1
        elif self.opinion[i]+self.opinion[j]==2:
            k=self.k2
        elif self.opinion[i]+self.opinion[j]==-2:
            k=self.k3
        else:
            k=-1
        assert k>-1,"breaking links probability Error!!!"
        if self.rdom(k):
            self.link.remove((i,j))
            self.adjacency[(i,j)]=0
            self.adjacency[(j,i)]=0
            if self.rdom(0.5):
                k=self.choose_node1(i)
                if k>i:
                    self.link.append((k,i))
                else:
                    self.link.append((i,k))
                self.adjacency[(k,i)]=1
                self.adjacency[(i,k)]=1
            else:
                k=self.choose_node1(j)
                if k>i:
                    self.link.append((k,j))
                else:
                    self.link.append((j,k))
                self.adjacency[(k,j)]=1
                self.adjacency[(j,k)]=1
                
    def voter(self):
        i=random.choice(range(self.N))
        j=self.choose_node2(i)
        if j!=-1:
            self.opinion[i]=self.opinion[j]
    
    def forward(self):
        self.stat=[]
        t=0
        while self.w**2<1 and len(self.stat)<1000000:
            self.stat.append([t,self.u,self.v,self.w])
            t+=1
            if self.rdom(self.phi):
                self.voter()
            else:
                self.linking()
            self.u=self.generate_u()
            self.v=self.generate_v()
            self.w=self.generate_w()
        self.stat=np.array(self.stat)
        #print("Run Time:",len(self.stat))
        


class IAdaptNetwork2:#The initial network is set up in advance.
    def __init__(self,k1,k2,k3,phi,opinion,adjacency):#type(opinion)=list or np.array,type(adjacency)=np.array
        self.k1=k1
        self.k2=k2
        self.k3=k3
        #k1=kd,k2=k3=ks
        self.phi=phi
        self.N=len(opinion)
        assert len(adjacency)==self.N,"opinion conflicts adjacency!!!"
        assert self.check_symmetric(adjacency),"adjacency matrix Error!!!"
        self.L=adjacency.sum()/2
        self.k=2*self.L/self.N
        self.opinion=np.array(opinion,dtype=np.int64)
        self.adjacency=np.array(adjacency,dtype=np.int64)     
        self.link=self.generate_link() #type(link)=list,storing 2-tuple         
        #self.N1=self.generate_A()
        #self.x=self.N1/self.N
        self.u=self.generate_u()
        #print("Initial u:",self.u)
        #print("u*=",self.fu())
        self.v=self.generate_v()
        self.w=self.generate_w()
        self.stat=[]
        
    def check_symmetric(self,a):
        if len(a)!=len(a[0]):
            return False
        return len((a.T==a)[(a.T==a)==False])==0
        
    def generate_link(self):
        link=[]
        for i in range(self.N):
            for j in range(i):#j<i
                if self.adjacency[i][j]==1:
                    link.append((i,j))
        assert len(link)==self.L,"link number error!!!"
        return link
        
    def generate_AA(self):
        AA=0
        for point in self.link:
            i,j=point
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==2:
                AA+=1
        return AA
    
    def generate_BB(self):
        BB=0
        for point in self.link:
            i,j=point
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==-2:
                BB+=1
        return BB
    
    def generate__AB(self):
        AB=0
        for point in self.link:
            i,j=point
            if self.adjacency[i][j]==1 and self.opinion[i]+self.opinion[j]==0:
                AB+=1
        return AB
    
    def generate_A(self):
        return len(self.opinion[self.opinion>0])
    
    def generate_B(self):
        return len(self.opinion[self.opinion<0])
    
    def generate_u(self):
        return (self.generate_AA()+self.generate_BB())/self.L
    
    def generate_v(self):
        return (self.generate_AA()-self.generate_BB())/self.L
    
    def generate_w(self):
        return (self.generate_A()-self.generate_B())/self.N
    
    def rdom(self,threshold):#0<threshold<1
        return random.random()<threshold
    
    def choose_node1(self,i):#No adjacent nodes
        total=[]
        for j in range(self.N):
            if self.adjacency[(i,j)]==0 and j!=i:
                total.append(j)
        assert len(total)>0,"Impossible!!!!"
        return random.choice(total)
    
    def choose_node2(self,i):#Adjacent nodes
        total=[]
        for j in range(self.N):
            if self.adjacency[(i,j)]==1:
                total.append(j)
        if len(total)==0:
            return -1
        return random.choice(total)
    
    def linking(self):
        i,j=random.choice(self.link)
        if self.opinion[i]+self.opinion[j]==0:
            k=self.k1
        elif self.opinion[i]+self.opinion[j]==2:
            k=self.k2
        elif self.opinion[i]+self.opinion[j]==-2:
            k=self.k3
        else:
            k=-1
        assert k>-1,"breaking links probability Error!!!"
        if self.rdom(k):
            self.link.remove((i,j))
            self.adjacency[(i,j)]=0
            self.adjacency[(j,i)]=0
            if self.rdom(0.5):
                k=self.choose_node1(i)
                if k>i:
                    self.link.append((k,i))
                else:
                    self.link.append((i,k))
                self.adjacency[(k,i)]=1
                self.adjacency[(i,k)]=1
            else:
                k=self.choose_node1(j)
                if k>i:
                    self.link.append((k,j))
                else:
                    self.link.append((j,k))
                self.adjacency[(k,j)]=1
                self.adjacency[(j,k)]=1
                
    def voter(self):
        i=random.choice(range(self.N))
        j=self.choose_node2(i)
        if j!=-1:
            self.opinion[i]=self.opinion[j]
    
    def forward(self):
        self.stat=[]
        t=0
        while self.w**2<1 and len(self.stat)<1000000:
            self.stat.append([t,self.u,self.v,self.w])
            t+=1
            if self.rdom(self.phi):
                self.voter()
            else:
                self.linking()
            self.u=self.generate_u()
            self.v=self.generate_v()
            self.w=self.generate_w()
        self.stat=np.array(self.stat)
        #print("Run Time:",len(self.stat))
        
def gen2(N,N1,k):#generate random undirected network
    opinion=np.array([1]*N1+[-1]*(N-N1),dtype=np.int64)
    mid1=np.array([0]*int(N1*(N-N1)),dtype=np.int64)
    mid1[random.sample(range(N1*(N-N1)),N*k//2)]=1
    adjacency=np.zeros((N,N),dtype=np.int64)
    i,j=N1,0
    for ii in range(len(mid1)):
        if j==N1:
            j=0
            i=i+1
        if mid1[ii]:
            adjacency[i][j]=mid1[ii]
            adjacency[j][i]=mid1[ii]
        j=j+1
    return opinion,adjacency

def alpha(phi,k,kd,ks):
    d=(2*phi*(3*k+1)+(1-phi)*(kd+ks))**2-16*phi*k*(2*phi*(k+1)+(1-phi)*kd)
    a=1/4-1/4/k-(1-phi)*(kd+ks)/8/phi/k+sqrt(d)/8/k/phi
    return d,a
def func(x,b):
    return 1-b*(1-x*x)
def r2(df):
    return r2_score(df.v,df.w)
