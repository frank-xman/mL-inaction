from numpy import *
def loadDataSet(filename):
	numFeat=len(open(filename).readline().split('\t')-1)
	dataMat=[];labelMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=[]
		curline=line.strip().split('\t')
		for i in range(numFeat);
			lineArr.append(float(curline[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curline[-1]))
	return dataMat,lableMat
def standRegres(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T
	xTx=xMat.T*xMat	
	if linalg.det(xTx)==0:
		print "this matrix is singular,cannot do inverse"
		return 
	ws=xTx.I*(xMat.T*yMat)
	return ws
def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat=mat(xArr);yMat=mat(yArr).T
	m=shape(xMat)[0]
	weight=mat(eye(m))
	for j in range(m):	
		diffMat=testPoint-xMat[j,:]
		weight[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx=xMat.T*(weight*xMat)
	if linalg.det(xTx)==0:
		print "this matrix cannot be inverse";
		return
	ws=xTx.I*xMat.T*weight*yMat
	return testPoint*ws
def lwlrTest(testArr,xArr,yArr,k=1.0):
	m=shape(testArr)[0]
	yHat=zeros(m)
	for i in range(m):
		yHat[i]=lwlr(testArr[i],xArr,yArr,k)
	return yHat

def ridgeRegres(xMat,yMat,lam=0.2):
	xTx=xMat.T*xMat
	denom=xTx+eye(shape(xMat)[1])*lam
	if linalg.det(denom)==0:
		print " u stupid fucking BAT "
	ws=denom.I*(xMat.T*yMat)
	return ws
def ridgeTest(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T	
	ymean=mean(yMat,0)
	yMat=yMat-ymean
	xmean=mean(xMat,0)
	xVar=var(xMat,0)
	xMat=(xMat-xmean)/xVar
	numiter=30
	wMat=zeros((numiter,shape(xMat)[1])
	for i in range(numiter):
		ws=ridgeRegres(xMat,yMat,exp(i-10))
		wMat[i,:]=ws.T
	return wMat
	
