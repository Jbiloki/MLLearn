#!/usr/bin/python3

import numpy as np
import pylab as pl
import mlp
import csv

def preprocessData(infile,outfile):
	fid = open(infile, 'r')
	oid = open(outfile, 'w')
	for s in fid:
		if(s.find('?')):
			s = s.replace('?', '0')
		oid.write(s)
	fid.close()
	oid.close()


def trainData():
	with open('fixeddata.csv') as f:
		numCol = csv.reader(f, delimiter=',')
		first_row = next(numCol)
		numCol = len(first_row)
		print(numCol)
		cancerData = np.loadtxt('fixeddata.csv',delimiter=',', usecols = range(1,11)).astype(np.float)
		print(np.shape(cancerData))
		cancerData = cancerData/cancerData.max(axis=0)
		np.random.shuffle(cancerData)
		
		traindata = cancerData[0:349, 0:9]
		traindatat = cancerData[0:349, 9:10]
		datavalid = cancerData[349:523, 0:9]
		datavalidt = cancerData[349:523, 9:10]
		datatest = cancerData[523:699, 0:9]
		datatestt = cancerData[523:699, 9:10]


		#net = mlp.mlp(traindata,traindatat,40,outtype='linear')
		#net.mlptrain(traindata,traindatat,.005,650)
		net = mlp.mlp(traindata,traindatat,10,outtype='linear')
		#net.mlptrain(traindata,traindatat,.4,1000)
		net.earlystopping(traindata,traindatat,datavalid,datavalidt,0.35)
		net.confmat(datatest,datatestt)

#preprocessData('breastcancerdata.data', 'fixeddata.csv')
trainData()
