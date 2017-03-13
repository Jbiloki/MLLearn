#!/usr/bin/python3

"""

Created on Sun Mar 12 2017


@author: Jacob

"""


def preprocessForrestFireData(infile,outfile):
	fid = open(infile, 'r')
	oid = open(outfile, 'w')
	firstLine = True
	for s in fid:
		if firstLine:
			firstLine = False
			oid.write(s)
			continue			
		s = s.replace('mon', '0')
		s = s.replace('tue', '1')
		s = s.replace('wed', '2')
		s = s.replace('thu', '3')
		s = s.replace('fri', '4')
		s = s.replace('sat', '5')
		s = s.replace('sun', '6')
		s = s.replace('jan', '0')
		s = s.replace('feb', '1')
		s = s.replace('mar', '2')
		s = s.replace('apr', '3')
		s = s.replace('may', '4')
		s = s.replace('jun', '5')
		s = s.replace('jul', '6')
		s = s.replace('aug', '7')
		s = s.replace('sep', '8')
		s = s.replace('oct', '9')
		s = s.replace('nov', '10')
		s = s.replace('dec', '11')
		oid.write(s)


	fid.close()
	oid.close()

import numpy as np
import pylab as pl
import mlp
#Process data once, uncomment to process
#preprocessForrestFireData('forestfires.csv', 'firesarranged.data')


def trainData():
	with open('firesarranged.data') as f:
		numCol = len(f.readline().split())
	fireData = np.loadtxt('firesarranged.data', dtype='S',delimiter=',',skiprows = 1, usecols = range(12)).astype(np.float)
	targets = np.loadtxt('firesarranged.data', dtype='S', delimiter=',', skiprows = 1, usecols = range(12,13)).astype(np.float)
	targets = np.reshape(targets,(517,1))
	#print(np.shape(fireData), np.shape(targets))
	#pl.plot(fireData,targets,'.')
	#pl.show()
	train = fireData[0::2,:]
	test = fireData[1::4,:]
	valid = fireData[3::4,:]
	
	traintarget = targets[0::2,:]
	testtarget = targets[1::4,:]
	validtarget = targets[3::4,:]
	

	
	net = mlp.mlp(train,traintarget,13,outtype='linear')
	#net.earlystopping(train,traintarget,valid,validtarget,0.1)
	#net.confmat(test,testtarget)
	net.mlptrain(train,traintarget,0.35,201)


trainData()
		
	




