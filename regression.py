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
preprocessForrestFireData('forestfires.csv', 'firesarranged.data')


def trainData():
	with open('firesarranged.data') as f:
		numCol = len(f.readline().split())
	fireData = np.loadtxt('firesarranged.data', dtype='S',delimiter=',',skiprows = 1, usecols = range(13)).astype(np.float)
	#fireData[:,:13] = fireData[:,:13]-fireData[:,:13].mean(axis=0)
	#print(np.shape(fireData), np.shape(targets))
	#imax = np.concatenate((fireData.max(axis=0)*np.ones((1,13)), np.abs(fireData.min(axis=0))*np.ones((1,13))), axis=0).max(axis=0)
	#targets[:,:1] = targets[:,:1]-targets[:,:1].mean(axis=0)
	#imax2 = np.concatenate((targets.max(axis=0)*np.ones((1)), np.abs(targets.min(axis=0))*np.ones((1))), axis=0).max(axis=0)
	#fireData[:,:13] = fireData[:,:13]/imax[:13]
	print(np.shape(fireData))
	print("Before", fireData)
	fireData = fireData/fireData.max(axis=0)
	print(fireData)
	#targets	[:,:1] = targets[:,:1]/imax2[:1]
	#print(np.shape(fireData), np.shape(targets))
	#pl.plot(fireData,targets,'.')
	#pl.show()
	np.random.shuffle(fireData)
	#print(imax)
	#print("Item", fireData.item(233), "Target", targets.item(233))
	#print(np.shape(fireData[:,:12]))
	train = fireData[0:258, 0:12]
	traintarget = fireData[0:258, 12:13]
	test = fireData[0:130,0:12]
	testtarget = fireData[0:130,12:13]
	valid = fireData[0:129,0:12]
	validtarget = fireData[0:129,12:13]
	

	
	#net = mlp.mlp(train,traintarget,105,outtype='linear')
	#net.mlptrain(train,traintarget,0.1,1001)
	net = mlp.mlp(train,traintarget,13,outtype='linear')
	net.mlptrain(train,traintarget,0.3,200)
	print(net.earlystopping(train,traintarget,valid,validtarget,0.1))
	#net.confmat(test,testtarget)
	


trainData()
		
	




