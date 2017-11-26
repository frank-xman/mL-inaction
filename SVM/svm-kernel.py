from numpy import *
import matplotlib.pyplot as plt
import random
class optStruct:
	def __init__(self,dataMatIn,classLabels,C,toler,kTup):
		self.X=dataMatIn
		self.labelMat=classLabels
		self.C=C
		self.tol=toler
		self.m=shape(dataMatIn)[0]
		self.alphas=mat(zeros((self.m,1)))
		self.b=0
		self.eCache=mat(zeros((self.m,2)))
		self.K=mat(zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)
def kernelTrans(X,A,kTup):
	m,n=shape(X)
	K=mat(zeros((m,1)))
	if kTup[0]=='lin':K=X*A.T
	elif kTup[0]=='rbf':
		for j in range(m):
			deltaRow=X[j,:]-A
			K[j]=deltaRow*deltaRow.T
		K=exp(K/(-1*kTup[1]**2))
	else: raise NameError('we have a problem')
	return K
	
def loadDataSet(filename):
	dataMat=[];labelMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat 
def calcEk(oS,k):
	fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
	Ek=fXk-float(oS.labelMat[k])
	return Ek
def selectJ(i,oS,Ei):
	maxK=-1;maxDeltaE=0;Ej=0
	oS.eCache[i]=[1,Ei]
	validEacheList=nonzero(oS.eCache[:,0].A)[0]	
	if(len(validEacheList))>1:
		for k in validEacheList:
			if k==i:continue
			Ek=calcEk(oS,k)
			deltaE=abs(Ei-Ek)	
			if(deltaE>maxDeltaE):
				maxK=k;maxDeltaE=deltaE;Ej=Ek
		return maxK,Ej
	else :
		j=selectJrand(i,oS.m)
		Ej=calcEk(oS,j)
	return j,Ej
def selectJrand(i,m):
	j=i
	while(j==i):
		j=int(random.uniform(0,m))
	return j
def updateEk(oS,k):
	Ek=calcEk(oS,k)
	oS.eCache[k]=[1,Ek]
def clipAlpha(aj,H,L):
	if aj>H:
		aj=H
	if L>aj:
		aj=L
	return aj
def innerL(i,oS):
	Ei=calcEk(oS,i)
	if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or \
		((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
		j,Ej=selectJ(i,oS,Ei)
		alphaIold=oS.alphas[i].copy()
		alphaJold=oS.alphas[j].copy()
		if(oS.labelMat[i]!=oS.labelMat[j]):
			L=max(0,oS.alphas[j]-oS.alphas[i])
			H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
		else:
			L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
			H=min(oS.C,oS.alphas[j]+oS.alphas[i])
		if L==H:
			print("L=H")
			return 0
		eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
			
		if eta>=0:
			print("eta>=0")
			return 0
		oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
		oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
		updateEk(oS,j)
		if(abs(oS.alphas[j]-alphaJold)<0.00001):	
			print("alpha_j change too small")
			return 0
		oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
		updateEk(oS,i)
		b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2=oS.b-Ej-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]
		if(0<oS.alphas[i])and(oS.C>oS.alphas[i]):oS.b=b1
		elif(0<oS.alphas[j])and(oS.C>oS.alphas[j]):oS.b=b2
		else:oS.b=(b1+b2)/2.0
		return 1
	else:	return 0
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
	oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
	iter=0
	entireSet=True;alphaPairsChanged=0	
	while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
		alphaPairsChanged=0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged+=innerL(i,oS)
				print ("the iter:%d i:%d ,pairschanged:%d"%\
					(iter,i,alphaPairsChanged))
			iter+=1
		else:
			nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
			for i in nonBoundIs:
				alphaPairsChanged+=innerL(i,oS)
				print ("non-bound,iter :%d  i:%d  ,pair changed\
					%d "%(iter,i,alphaPairsChanged))
			iter+=1
		if entireSet:entireSet=False
		elif(alphaPairsChanged==0):entireSet=True
		print"iteration number: %d"% iter
	return oS.b,oS.alphas







def testRbf(k1=1.3):
	dataArr,labelArr=loadDataSet('testSetRBF.txt')
	b,alphas=smoP(dataArr,labelArr,200,0.0001,100,('rbf',k1))
	datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
	svInd=nonzero(alphas.A>0)[0]
	sVs=datMat[svInd]
	labelSV=labelMat[svInd]
	print "there are %d Support vector"%shape(sVs)[0]
	m,n=shape(datMat)
	errorCount=0
	for i in range(m):
		kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1))
		predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
		if sign(predict)!=sign(labelArr[i]):errorCount+=1
	print "the training error rate is : %.2f%%"%(float(errorCount)/m*100)
	#showClassifer(datMat,labelMat,alphas)
	dataArr,labelArr=loadDataSet('testSetRBF2.txt')
	errorCount=0
	datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
	m,n=shape(datMat)
	for i in range(m):	
		kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1))
		predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
		if sign(predict)!=sign(labelArr[i]):errorCount+=1
	print " the test error rate is : %.2f%%"%(float(errorCount)/m*100)
	showClassifer(datMat,labelMat,alphas)
def showClassifer(dataMat,classLabels,alphas):
	
	data_plus=[]
	data_minus=[]
	for i in range(len(dataMat)):
		if classLabels[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus=array(data_plus)
	data_minus=array(data_minus)
	plt.scatter(transpose(data_plus)[0],transpose(data_plus)[1],s=30,alpha=0.7,marker='^')
	plt.scatter(transpose(data_minus)[0],transpose(data_minus)[1],s=30,alpha=0.7,marker='s')
#	x1=max(dataMat)[0]
#	x2=min(dataMat)[0]
#	a1,a2=w
#	b=float(b)
#	a1=float(a1[0])
#	a2=float(a2[0])
#	y1,y2=(-b-a1*x1)/a2,(-b-a1*x2)/a2
#	plt.plot([x1,x2],[y1,y2])a

#	for i,alpha in enumerate(alphas):
#		if abs(alpha)>0:
#			x=dataMat[i]
#			
#			plt.scatter([x],[y],s=150,c='none',alpha=0.7,\
#			linewidth=1.5,edgecolor='red')
	plt.show()
if __name__=='__main__':
#	dataArr,classLabels=loadDataSet('testSetRBF.txt')
#	b,alphas=smoP(dataArr,classLabels,0.6,0.001,40)
#	w=calcWs(alphas,dataArr,classLabels)
#	showClassifer(dataArr,classLabels,w,b)
	testRbf()
#	showClassifer(dataArr,classLabels)
