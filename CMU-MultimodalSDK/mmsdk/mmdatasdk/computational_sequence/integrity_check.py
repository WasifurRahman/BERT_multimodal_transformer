from mmsdk.mmdatasdk import log
from mmsdk.mmdatasdk.configurations.metadataconfigs import *
from tqdm import tqdm

#this function checks the heirarchy format of a given computatioanl sequence data. This will crash the program if data is in wrong format. If in correct format the return value is simply True 
def validateDataIntegrity(data,rootName,which=True):
	log.status("Checking the integrity of the data in <%s> computational sequence ..."%rootName)

	pbar = tqdm(total=len(data.keys()),unit=" Computational Sequence Entries",leave=False)
	failure=False
	if (type(data) is not dict):
		#this will cause the rest of the pipeline to crash - RuntimeError
		log.error("%s computational sequence data is not in heirarchy format ...",error=True)
	try:
		#for each video check the shapes of the intervals and features
		for vid in data.keys():
			#check the intervals first - if failure simply show a warning - no exit since we want to identify all the cases
			if len(data[vid]["intervals"].shape) != 2 :
				if which: log.error("Video <%s> in  <%s> computational sequence has wrong intervals array shape. "%(vid,rootName),error=False)
				failure=True
			#check the features next
			if len(data[vid]["features"].shape) != 2 :
				if which: log.error("Video <%s> in  <%s> computational sequence has wrong features array shape. "%(vid,rootName),error=False)
				failure=True
			#if the first dimension of intervals and features doesn't match
			if data[vid]["features"].shape[0] != data[vid]["intervals"].shape[0]:
				if which: log.error("Video <%s> in <%s> computational sequence - features and intervals have different first dimensions. "%(vid,rootName),error=False)
				failure=True
			pbar.update(1)
	#some other thing has happened! - RuntimeError
	except:
		if which:
			log.error("<%s> computational sequence data itegrity could not be checked. "%rootName,error=True)
		pbar.close()
	pbar.close()

	#failure during intervals and features check
	if failure:
		log.error("<%s> computational sequence data integrity check failed due to inconsistency in intervals and features. "%rootName,error=True)
	else:
		log.success("<%s> computational sequence data in correct format." %rootName)
		return True


#this function checks the computatioanl sequence metadata. This will crash the program if metadata is missing. If metadata is there the return value is simply True 
def validateMetadataIntegrity(metadata,rootName,which=True):
	log.status("Checking the integrity of the metadata in <%s> computational sequence ..."%rootName)
	failure=False
	if type(metadata) is not dict:
		log.error("<%s> computational sequence metadata is not key-value pairs!", error=True)
	presenceFlag=[mtd in metadata.keys() for mtd in featuresetMetadataTemplate]
	#check if all the metadata is set
	if all (presenceFlag) is False:
		#which one is not set
		if which:
			missings=[x for (x,y) in zip (featuresetMetadataTemplate,presenceFlag) if y is False]
			log.error("Missing metadata in <%s> computational sequence: %s"%(rootName,str(missings)),error=False)
		failure=True
		#if failed before
	if failure:
		log.error(msgstring="<%s> computational sequence does not have all the required metadata ..."%rootName,error=True)
	else:
		log.success("<%s> computational sequence metadata in correct format"%rootName)
	return True


