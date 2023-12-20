path="D://Users/1/Desktop/dataset/dataset66/"
phis=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
df_t=pd.read_excel(path+'result_t.xlsx')
df_p=pd.read_excel(path+'result_p.xlsx')
df_r=pd.read_excel(path+'result_r.xlsx')
ddf_t=df_t.iloc[:,1:]
ddf_p=df_p.iloc[:,1:]
ddf_r=df_r.iloc[:,1:]
index=np.array(df_t.iloc[:,0])
ddf_t.index=index
ddf_p.index=index
ddf_r.index=index
from math import *
fig,ax=plt.subplots(dpi=300,figsize=(5,4))
phis=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
k,kd,ks=8,0.6,0.6
phi=np.linspace(0.05,1,1000)
y=list(map(lambda x:((10000-50)*log(99)-10000*log(50))/2/alpha(x,k,kd,ks)[1]/x,phi))
plt.plot(phi,y,label='Theory',color='r')
'''
th=[]
for phi in phis:
    th.append(((10000-50)*log(99)-10000*log(50))/2/alpha(phi,k,kd,ks)[1]/phi)
plt.plot(phis,np.array(th),color='r')
'''
plt.scatter(phis,np.array(ddf_t.loc[['mean']].T),color='k',label='Simulation')
plt.xlim(-0.01,1.01)
yticks=list(range(0, int(max(y)+30000), 30000))
ax.set_yticks(yticks)
ax.set_yticklabels([f"{int(y/10000)}" for y in yticks])
text = r"$\times 10^4$"
ax.text(0.01, 1.05, text, transform=ax.transAxes, verticalalignment='top')

plt.xlabel("$\phi$",fontsize=15)
plt.ylabel("Fixation time $T_{N/2}$",fontweight='bold')
plt.xticks(np.arange(0,1.1,0.1))
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('D://Users/1/Desktop/image/benchmark_FT/'+'benchtime4.svg')
plt.show()
