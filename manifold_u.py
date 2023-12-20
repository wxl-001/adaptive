#manifold
path="D://Users/1/Desktop/dataset/dataset66/"
df=pd.read_excel(path+'50_25.xlsx')
w=np.linspace(-1,1,1000)
phis=[0.5]
fig,ax=plt.subplots(dpi=300,figsize=(5,4))
plt.plot(df.w,df.u,color='b',label="Simulation")
phi,k,kd,ks=phis[0],8,0.6,0.6
#plt.plot(w,list(map(lambda x:uv(kd,ks,k,phi,x)[1],w)),color='r',label='Numerical')
a=alpha(phi,k,kd,ks)[1]
plt.plot(w,list(map(lambda x:1-a*(1-x*x),w)),color='k',label='Theory')
n=len(df)
popt, pcov = curve_fit(func, df.w[n//10:], df.u[n//10:])
y_pred = [func(i, popt[0]) for i in w]
plt.plot(w,y_pred,color='g',label='Fit')
print(n,1-a,1-popt[0])

plt.ylim(0,1)
plt.xlim(-1,1)
plt.xticks([-1,-0.5,0,0.5,1])
plt.xlabel('w',fontsize=15)
plt.ylabel('u',fontsize=15)

plt.annotate('alpha={}\nfitting={}'.format(round(a,3),round(popt[0],3)),
             xy=(0,1-a),
             xytext=(-0.8,0.3),
             fontsize=12,
             bbox={'facecolor': '#74C476', #填充色
              'edgecolor':'b',#外框色
               'alpha':0, #框透明度
               'pad': 0.8,#本文与框周围距离 
               'boxstyle':'round4'
              },
             arrowprops=dict(facecolor='#74C476',
                             alpha=0.6,
                             arrowstyle='-|>',
                             connectionstyle='arc3,rad=0.5',
                             
                            )
            )

plt.legend(loc=4,fontsize=13)
plt.tight_layout()
#plt.savefig('D://Users/1/Desktop/PRE/image/manifold/'+'50_25u1.svg')
plt.show()
