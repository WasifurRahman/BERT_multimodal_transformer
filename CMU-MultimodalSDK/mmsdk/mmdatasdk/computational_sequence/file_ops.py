import sys
import h5py
import os
from tqdm import tqdm
from mmsdk.mmdatasdk import log
from mmsdk.mmdatasdk.configurations.metadataconfigs import *
from mmsdk.mmdatasdk.computational_sequence.integrity_check import *

#reading MTD files directly from HDD
def readCSD(resource,destination=None):

	if (resource is None): raise log.error("No resource specified for computational sequence!",error=True)	
	if os.path.isfile(resource) is False:
		log.error("%s file not found, please check the path ..."%resource,error=True)	
	try:
		h5handle=h5py.File('%s'%resource,'r')
	except: 
		raise log.error("%s resource is not a valid hdf5 computational sequence  ..."%resource,error=True)
	log.success ("Computational sequence read from file %s ..."%resource)
	return h5handle,dict(h5handle[list(h5handle.keys())[0]]["data"]),metadataToDict(h5handle[list(h5handle.keys())[0]]["metadata"])
	

#writing MTD files to disk
def writeCSD(data,metadata,rootName,destination):
	#check the data to make sure it is in correct format
	validateDataIntegrity(data,rootName)
	validateMetadataIntegrity(metadata,rootName)

	log.status("Writing the <%s> computational sequence data to %s"%(rootName,destination))	
	#opening the file
	writeh5Handle=h5py.File(destination,'w')
	#creating the root handle
	rootHandle=writeh5Handle.create_group(rootName)

	#writing the data
	dataHandle=rootHandle.create_group("data")
	pbar = tqdm(total=len(data.keys()),unit=" Computational Sequence Entries",leave=False)
	for vid in data:
		vidHandle=dataHandle.create_group(vid)
		vidHandle.create_dataset("features",data=data[vid]["features"])
		vidHandle.create_dataset("intervals",data=data[vid]["intervals"])
		pbar.update(1)
	pbar.close()
	log.success("<%s> computational sequence data successfully wrote to %s"%(rootName,destination))
	log.status("Writing the <%s> computational sequence metadata to %s"%(rootName,destination))
	#writing the metadata
	metadataHandle=rootHandle.create_group("metadata")
	for metadataKey in metadata.keys():
		metadataHandle.create_dataset(metadataKey,(1,),dtype=h5py.special_dtype(vlen=unicode) if sys.version_info.major is 2 else h5py.special_dtype(vlen=str))
		cast_operator=unicode if sys.version_info.major is 2 else str
		metadataHandle[metadataKey][0]=cast_operator(metadata[metadataKey])

	writeh5Handle.close()
	log.success("<%s> computational sequence metadata successfully wrote to %s"%(rootName,destination))
	log.success("<%s> computational sequence successfully wrote to %s ..."%(rootName,destination))

def metadataToDict(mtdmetadata):
	if (type(mtdmetadata) is dict): 
		return mtdmetadata
	else:
		metadata={}
		for key in mtdmetadata.keys(): 
			metadata[key]=mtdmetadata[key][0] 
		return metadata


