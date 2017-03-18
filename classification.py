#!/usr/bin/python3

import numpy as np
import pylab as pl
import mlp

def preprocessData(infile,outfile):
	fid = open(infile, 'r')
	oid = open(outfile, 'w')
	for s in fid:
		if(s.find('?')):
			s = s.replace('?,', '')
		oid.write(s)
	fid.close()
	oid.close()


def trainData():
	with open('fixeddata.data') as f:
		numCol = len(f.readline())
		print(numCol)
	cancerData = np.loadtxt('fixeddata.data',delimiter=',', dtype='S',usecols= range(0,11)).astype(np.float)
	print(np.shape(cancerData))

preprocessData('breastcancerdata.data', 'fixeddata.data')
trainData()
