#generate manifold v
path="D://Users/1/Desktop/dataset/dataset66/"
df=pd.read_excel(path+'50_25.xlsx')
w=np.linspace(-1,1,1000)
phis=[0.5]
fig,ax=plt.subplots(dpi=300,figsize=(5,4))
plt.plot(df.w,df.v,color='b',label="Simulation")
phi,k,kd,ks=phis[0],8,1,1
#plt.plot(w,list(map(lambda x:uv(kd,ks,k,phi,x)[1],w)),color='r',label='Numerical')
a=alpha(phi,k,kd,ks)[1]
plt.plot(w,w,color='k',label='Theory')
n=len(df)

plt.ylim(-1,1)
plt.xlim(-1,1)
plt.xticks([-1,-0.5,0,0.5,1])
plt.yticks([-1,-0.5,0,0.5,1])
plt.xlabel('w',fontsize=15)
plt.ylabel('v',fontsize=15)
plt.legend(loc=4,fontsize=13)
plt.tight_layout()
#plt.savefig('D://Users/1/Desktop/PRE/image/manifold/'+'50_25v1.svg')
plt.show()
