import h5py
import hashlib
import validators
import json
import sys
import os 
import time
import uuid
from mmsdk.mmdatasdk import log 
from mmsdk.mmdatasdk.configurations.metadataconfigs import *
from mmsdk.mmdatasdk.computational_sequence.integrity_check import *
from mmsdk.mmdatasdk.computational_sequence.blank import *
from mmsdk.mmdatasdk.computational_sequence.file_ops import *
from mmsdk.mmdatasdk.computational_sequence.download_ops import *

#computational sequence class
#main attributes:
#       mainFile: where the location of the heirarchical data binary file is
#       resource: where the heirarchical data comes from (i.e. a URL)
#       h5handle: handle to the file
#       data: the data in a dictionary format
#       metadata: the metadata in a dictionary format
#       
#main function:
#       completeAllMissingMetadata: a helper to complete all the missing metadata
#       deploy: deploy the computational sequence - you can also notify multicomp about your computational sequence 

class computational_sequence():

	def __init__(self,resource, destination=None, validate=True,noisy=True):
		#initializing the featureset
		h5handle,data,metadata=self._initialize(resource,destination)
		self.h5handle=h5handle
		#initializing based on pre-existing computational sequence - only if h5handle is None 
		if self.h5handle is not None:
			if type(metadata) is dict and "root name" in metadata.keys():
				self.rootName=metadata["root name"]
			else:
				self.rootName=resource
			self.setData(data,self.rootName)
			self.setMetadata(metadata,self.rootName)
		else:
			self.data=data
			self.metadata=metadata

	def _compare_entries(self,entry1,entry2):
		return entry1.split('[')[0]==entry2.split('[')[0]

	def __getitem__(self, key):
		return self.data[key]
	
	def keys(self):
		return self.data.keys()

	def _remove_id(self,entry_id):
		if entry_id in list(self.data.keys()):
			del self.data[entry_id]

	#TODO: try and excepts to be added to this code
	def _initialize(self,resource,destination):
		#computational sequence is already initialized
		if hasattr(self,'h5handle'): raise log.error("<%s> computational sequence already initialized ..."%self.metadata["root name"],error=True)
		#initialization type
		optype=None
		#initializing blank - mainFile is where to initialize the data and resource is None since the data comes from nowhere
		if '.csd' not in resource:
			self.mainFile=None
			#self.resource will be None since there is nowhere this was read from - resource being passed to initBlank is the name of root
			self.resource=None
			self.rootName=resource
			return initBlank(self.rootName)
		#reading from url - mainFile is where the data should go and resource is the url
		else:
	#	try:
			if validators.url(resource):
				#user would like to store to the current directory
				if destination is None or destination == '':
					destination=os.path.join('./',resource.split('/')[-1])
				#user has chosen a different directory
				elif '.csd' not in destination:
					destination=os.path.join(destination,resource.split('/')[-1])
				readURL(resource,destination)
				self.mainFile=destination
				self.resource=resource
			else:
				self.mainFile=resource
			return readCSD(self.mainFile)
	

	#checking if the data and metadata are in correct format
	#stops the program if the integrity is not ok
	def _checkIntegrity(self,error=True):
		if not hasattr(self,'metadata') or not hasattr(self,'data'):
			log.error("computational sequence is blank (data or metadata is missing)")
		log.status("Checking the integrity of the <%s> computational sequence ..."%self.metadata["root name"])
		#TODO: hash check not implemented yet
		datavalid=validateDataIntegrity(self.data,self.metadata["root name"],which=False)
		metadatavalid=validateMetadataIntegrity(self.metadata,self.metadata["root name"],which=False)
		if datavalid and metadatavalid:
			log.success("<%s> computational sequence is valid!"%self.metadata["root name"])


	#set the metadata for all the missing information in the metadata. If the key is not set then the assumption is that metadata is not available. Note that if the metadata is set to a dummy variable like None then it is still considered set.
	def completeAllMissingMetadata(self):
		
		missings=[x for (x,y) in zip(featuresetMetadataTemplate,[metadata in self.metadata.keys() for metadata in featuresetMetadataTemplate]) if y is False]
		#python2 vs python 3
		#TODO: Add read from file
		root_name_ext=''
		if hasattr(self,"rootName"):
			root_name_ext=" for <%s> computational sequence"%self.rootName
		for missing in missings:
			if sys.version_info.major is 2:
				self.metadata[missing]=raw_input("Please input %s%s: "%(missing,root_name_ext))
			
			if sys.version_info.major is 3:
				self.metadata[missing]=input("Please input %s%s: "%(missing,root_name_ext))

	def setData(self,data,rootName):
		validateDataIntegrity(data,rootName,which=True)
		self.data=data

	def setMetadata(self,metadata,rootName):
		validateMetadataIntegrity(metadata,rootName,which=False)
		self.metadata=metadata

	#writing the file to the output
	def deploy(self,destination):
		self.completeAllMissingMetadata()
		self._checkIntegrity()
		log.status("Deploying the <%s> computational sequence to %s"%(destination,self.metadata['root name']))
		#generating the unique identifiers
		self.metadata['uuid']=uuid.uuid4()
		#TODO: add SHA256 check + midification should not be possible without private key
		self.metadata['md5']=None
		log.status("Your unique identifier for <%s> computational sequence is %s"%(self.metadata["root name"],self.metadata['uuid']))
		writeCSD(self.data,self.metadata,self.metadata["root name"],destination)
		self.mainFile=destination

	def _get_entries_stripped(self):
		return list(set([entry.split('[')[0] for entry in list(self.data.keys())]))
		

	def bib_citations(self,outfile=None):
		outfile=sys.stdout if outfile is None else outfile
		if self.metadata is None or self.metadata=={}:
			log.error("Metadata is not set for <%s> computational sequence"%self.rootName)
		outfile.write('Computational Sequence <%s> bib: '%self.rootName+self.metadata['featureset bib citation']+'\n\n')
		outfile.write('Dataset <%s> bib: '%self.metadata["dataset name"]+self.metadata['dataset bib citation']+'\n\n')

