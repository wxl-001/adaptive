#statistical data
path="D://Users/1/Desktop/dataset/dataset63p/"#file load path
phis=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
list1=[]
list4=[]
list5=[]
kd,ks=0.6,0.3
for phi in phis:
    list2=[]
    list3=[]
    list6=[]
    for i in range(50):
        if phi*100<10:
            a='0'+str(int(phi*100))
        else:
            a=str(int(phi*100))
        if i+1<10:
            c='0'+str(i+1)
        else:
            c=str(i+1)
        s=a+'_'+c
        path1=path+s+'.xlsx'
        #generate and save data
        '''
        NN=IAdaptNetwork1(kd,ks,ks,phi,400,100,20)
        NN.forward()
        df=pd.DataFrame(NN.stat[:,1:4],columns=['u','v','w'])
        df.to_excel(path1)
        '''
        #start
        df=pd.read_excel(path1)#read data
        list2.append(np.array(df.w)[-1]>0)
        list3.append(len(df))
        list6.append(r2(df))
        #end  
    list1.append(list2)#mean=prob
    list4.append(list3)#time
    list5.append(list6)#R2
l1=[sum(i)/50 for i in list1]#prob
l2=[sum(i)/50 for i in list4]#time
l3=[sum(i)/50 for i in list5]#R2
dff=pd.DataFrame([l1,l2,l3])
dff.index=['Prob','Time','R2']
dff.columns=phis
dff.to_excel(path+'result.xlsx')
