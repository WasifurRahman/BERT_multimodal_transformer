from mmsdk.mmdatasdk import log


def initBlank(resource):	
	data={}
	metadata={}
	metadata["root name"]=resource
	log.success("Initialized empty <%s> computational sequence."%metadata["root name"])
	return None,data,metadata


