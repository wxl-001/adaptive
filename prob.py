from matplotlib.ticker import MultipleLocator, FormatStrFormatter
path36="D://Users/1/Desktop/dataset/PT/phi3/data66/"
path66="D://Users/1/Desktop/dataset/PT/phi6/data66/"

# path69 denotes the case where phi=0.6 ks=0.9.
# kd=0.6 satisfies all cases.
N1s=[0,10,20,30,40,50,60,70,80,90,100]

df36=pd.read_excel(path36+'result.xlsx')

df66=pd.read_excel(path66+'result.xlsx')


fig,ax=plt.subplots(dpi=300,figsize=(5,4))
#plt.figure(dpi=300,figsize=(8,4))
y=np.array(list(map(lambda x:0.01*x,N1s)))
plt.xlabel('The initial number of the opinion $A$',fontweight='bold')
plt.ylabel('Fixation probability of $A$',fontweight='bold')
ax.xaxis.set_major_locator(MultipleLocator(20))

ax.xaxis.set_minor_locator(MultipleLocator(10))

plt.plot(N1s,y,'k')
#plt.plot(N1s,np.array([0]+list(df33.iloc[0][1:])+[1]),'s-',label='$\phi,kd,ks=0.3,0.6,0.3$')
#plt.plot(N1s,np.array([0]+list(df63.iloc[0][1:])+[1]),'s-',label='$\phi,kd,ks=0.6,0.6,0.3$')
#plt.plot(N1s,np.array([0]+list(df36.iloc[0][1:])+[1]),'s-')
#kd,ks=0.6,0.9
plt.plot(N1s,np.array([0]+list(df36.iloc[0][1:])+[1]),'o-',label='$\phi=0.3$')

#plt.plot(N1s,np.array([0]+list(df66.iloc[0][1:])+[1]),'s-')
plt.plot(N1s,np.array([0]+list(df66.iloc[0][1:])+[1]),'s-',label='$\phi=0.6$')
plt.legend(loc=4)
#path="D://Users/1/Desktop/dataset/PT/image/"
plt.tight_layout()
#plt.savefig('D://Users/1/Desktop/image/benchmark_FP/'+'benchprob4.svg')
plt.show()
