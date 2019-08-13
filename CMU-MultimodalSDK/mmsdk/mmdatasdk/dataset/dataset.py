from mmsdk.mmdatasdk import log, computational_sequence
import sys
import numpy
import time
from tqdm import tqdm
import os

epsilon=10e-4

class mmdataset:

	def __init__(self,recipe,destination=None):
		self.computational_sequences={}

		if type(recipe) is str:
			if os.path.isdir(recipe) is False:
				log.error("Dataset folder does not exist ...",error=True)

			from os import listdir
			from os.path import isfile, join
			computational_sequence_list = [f for f in listdir(recipe) if isfile(join(recipe, f)) and f[-4:]=='.csd']
			for computational_sequence_fname in computational_sequence_list:
				this_sequence=computational_sequence(join(recipe,computational_sequence_fname))
				self.computational_sequences[this_sequence.metadata["root name"]]=this_sequence

		if type(recipe) is dict:
			for entry, address in recipe.items():
				self.computational_sequences[entry]=computational_sequence(address,destination)

		if len(self.computational_sequences.keys())==0:
			log.error("Dataset failed to initialize ...", error=True)

		log.success("Dataset initialized successfully ... ")

	def __getitem__(self,key):
		if key not in list(self.computational_sequences.keys()):
			log.error("Computational sequence does not exist ...",error=True)
		return self.computational_sequences[key]
	
	def keys(self):
		return self.computational_sequences.keys()

	def add_computational_sequences(self,recipe,destination):
		for entry, address in recipe.items():
			if entry in self.computational_sequences:
				log.error("Dataset already contains <%s> computational sequence ..."%entry)
			self.computational_sequences[entry]=computational_sequence(address,destination)

	def bib_citations(self,outfile=None):
		outfile=sys.stdout if outfile is None else outfile
		sdkbib='@article{zadeh2018multi, title={Multi-attention recurrent network for human communication comprehension}, author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe}, journal={arXiv preprint arXiv:1802.00923}, year={2018}}'
		outfile.write('mmsdk bib: '+sdkbib+'\n\n')
		for entry,compseq in self.computational_sequences.items():
			compseq.bib_citations(outfile)

	def __unify_dataset(self,active=True):
		log.status("Unify was called ...")


		all_vidids={}
		violators=[]

		

		for seq_key in list(self.computational_sequences.keys()):
			for vidid in list(self.computational_sequences[seq_key].data.keys()):
				vidid=vidid.split('[')[0]
				all_vidids[vidid]=True

		for vidid in list(all_vidids.keys()):
			for seq_key in list(self.computational_sequences.keys()):
				if not any([vidid_in_seq for vidid_in_seq in self.computational_sequences[seq_key].data.keys() if vidid_in_seq[:len(vidid)]==vidid]):
					violators.append(vidid)
		if len(violators) >0 :
			for violator in violators:
				log.error("%s entry is not shared among all sequences, removing it ..."%violator,error=False)
				if active==True:
					self.__remove_id(violator)
		if active==False and len(violators)>0:
			log.error("%d violators remain, alignment will fail if called ..."%len(violators),error=True)

		log.success("Unify finished, dataset is compatible for alignment ...")


	def __remove_id(self,entry_id):
		for _,compseq in self.computational_sequences.items():
			compseq._remove_id(entry_id)


	def align(self,reference,collapse_functions=None,replace=True):
		aligned_output={}

		for sequence_name in self.computational_sequences.keys():
			aligned_output[sequence_name]={}
		if reference not in self.computational_sequences.keys():
			log.error("Computational sequence <%s> does not exist in dataset"%reference,error=True)
		refseq=self.computational_sequences[reference].data
		#unifying the dataset, removing any entries that are not in the reference computational sequence
		self.__unify_dataset()

		#building the relevant entries to the reference - what we do in this section is simply removing all the [] from the entry ids and populating them into a new dictionary
		log.status("Alignment based on <%s> computational sequence will start shortly ..."%reference)
		relevant_entries=self.__get_relevant_entries(reference)

		pbar = tqdm(total=len(refseq.keys()),unit=" Computational Sequence Entries",leave=False)
		pbar.set_description("Overall Progress")
		for entry_key in list(refseq.keys()):
			pbar_small=tqdm(total=refseq[entry_key]['intervals'].shape[0],unit=" Segments",leave=False)
			pbar_small.set_description("Aligning %s"%entry_key)
			for i in range(refseq[entry_key]['intervals'].shape[0]):
				#interval for the reference sequence
				ref_time=refseq[entry_key]['intervals'][i,:]
				#we drop zero or very small sequence lengths - no align for those
				if (abs(ref_time[0]-ref_time[1])<epsilon):
					pbar_small.update(1)
					continue

				#aligning all sequences (including ref sequence) to ref sequence
				for otherseq_key in list(self.computational_sequences.keys()):
					if otherseq_key != reference:
						intersects,intersects_features=self.__intersect_and_copy(ref_time,relevant_entries[otherseq_key][entry_key],epsilon)
					else:
						intersects,intersects_features=refseq[entry_key]['intervals'][i,:][None,:],refseq[entry_key]['features'][i,:][None,:]
					#there were no intersections between reference and subject computational sequences for the entry
					if intersects.shape[0] == 0:
						continue
					#collapsing according to the provided functions
					if type(collapse_functions) is list:
						intersects,intersects_features=self.__collapse(intersects,intersects_features,collapse_functions)
					if(intersects.shape[0]!=intersects_features.shape[0]):
						log.error("Dimension mismatch between intervals and features when aligning <%s> computational sequences to <%s> computational sequence"%(otherseq_key,reference))
					aligned_output[otherseq_key][entry_key+"[%d]"%i]={}
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"]=intersects
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["features"]=intersects_features
				pbar_small.update(1)
			pbar_small.close()
			pbar.update(1)
		pbar.close()
		log.success("Alignment to <%s> complete."%reference)
		if replace is True:
			log.status("Replacing dataset content with aligned computational sequences")
			self.__set_computational_sequences(aligned_output)
			return None
		else:
			log.status("Creating new dataset with aligned computational sequences")
			newdataset=mmdataset({})
			newdataset.__set_computational_sequences(aligned_output,metadata_copy=False)
			return newdataset

	def __collapse(self,intervals,features,functions):
		#we simply collapse the intervals to (1,2) matrix
		new_interval=numpy.array([[intervals.min(),intervals.max()]])
		try:
			new_features=numpy.concatenate([function(intervals,features) for function in functions],axis=0)
			if len(new_features.shape)==1:
				new_features=new_features[None,:]
		except:
			log.error("Cannot collapse given the set of function.", error=True)
		return new_interval,new_features


	#setting the computational sequences in the dataset based on a given new_computational_sequence_data - may copy the metadata if there is already one
	def __set_computational_sequences(self,new_computational_sequences_data,metadata_copy=True):
	
		#getting the old metadata from the sequence before replacing it. Even if this is a new computational sequence this will not cause an issue since old_metadat will just be empty
		old_metadata={m:self.computational_sequences[m].metadata for m in list(self.computational_sequences.keys())}
		self.computational_sequences={}
		for sequence_name in list(new_computational_sequences_data.keys()):
			self.computational_sequences[sequence_name]=computational_sequence(sequence_name)
			self.computational_sequences[sequence_name].setData(new_computational_sequences_data[sequence_name],sequence_name)
			if metadata_copy:
				#if there is no metadata for this computational sequences from the previous one or no previous computational sequenece
				if sequence_name not in list(old_metadata.keys()):
					log.error ("Metadata not available to copy ..., please provide metadata before writing to disk later", error =False)
				self.computational_sequences[sequence_name].setMetadata(old_metadata[sequence_name],sequence_name)
			self.computational_sequences[sequence_name].rootName=sequence_name

	def deploy(self,destination,filenames):
		if os.path.isdir(destination) is False:
			os.mkdir(destination)
		for seq_key in list(self.computational_sequences.keys()):
			if seq_key not in list(filenames.keys()):
				log.error("Filename for %s computational sequences not specified"%seq_key)
			filename=filenames[seq_key]
			if filename [:-4] != '.csd':
				filename+='.csd'
			self.computational_sequences[seq_key].deploy(os.path.join(destination,filename))

	def __intersect_and_copy(self,ref,relevant_entry,epsilon):

		sub=relevant_entry["intervals"]
		features=relevant_entry["features"]

		#copying and inverting the ref
		ref_copy=ref.copy()
		ref_copy[1]=-ref_copy[1]
		ref_copy=ref_copy[::-1]
		sub_copy=sub.copy()
		sub_copy[:,0]=-sub_copy[:,0]
		#finding where intersect happens
		where_intersect=(numpy.all((sub_copy-ref_copy)>(-epsilon),axis=1)==True)
		intersectors=sub[where_intersect,:]
		intersectors=numpy.concatenate([numpy.maximum(intersectors[:,0],ref[0])[:,None],numpy.minimum(intersectors[:,1],ref[1])[:,None]],axis=1)
		intersectors_features=features[where_intersect,:]
		#checking for boundary cases and also zero length
		where_nonzero_len=numpy.where(abs(intersectors[:,0]-intersectors[:,1])>epsilon)
		intersectors_final=intersectors[where_nonzero_len]
		intersectors_features_final=intersectors_features[where_nonzero_len]
		return intersectors_final,intersectors_features_final

	#TODO: Need tqdm bar for this as well
	def __get_relevant_entries(self,reference):
		relevant_entries={}
		relevant_entries_np={}

		#pbar = tqdm(total=count,unit=" Computational Sequence Entries",leave=False)


		for otherseq_key in set(list(self.computational_sequences.keys()))-set([reference]):
			relevant_entries[otherseq_key]={}
			relevant_entries_np[otherseq_key]={}
			sub_compseq=self.computational_sequences[otherseq_key] 
			for key in list(sub_compseq.data.keys()):              
				keystripped=key.split('[')[0]                  
				if keystripped not in relevant_entries[otherseq_key]:                           
					relevant_entries[otherseq_key][keystripped]={}
					relevant_entries[otherseq_key][keystripped]["intervals"]=[]                     
					relevant_entries[otherseq_key][keystripped]["features"]=[]                                                            
		        
				relev_intervals=self.computational_sequences[otherseq_key].data[key]["intervals"]                                             
				relev_features=self.computational_sequences[otherseq_key].data[key]["features"]         
				if len(relev_intervals.shape)<2:
					relev_intervals=relev_intervals[None,:]
					relev_features=relev_features[None,:]

				relevant_entries[otherseq_key][keystripped]["intervals"].append(relev_intervals)
				relevant_entries[otherseq_key][keystripped]["features"].append(relev_features)
		                        
			for key in list(relevant_entries[otherseq_key].keys()):
				relev_intervals_np=numpy.concatenate(relevant_entries[otherseq_key][key]["intervals"],axis=0)                                 
				relev_features_np=numpy.concatenate(relevant_entries[otherseq_key][key]["features"],axis=0)
				sorted_indices=sorted(range(relev_intervals_np.shape[0]),key=lambda x: relev_intervals_np[x,0])                               
				relev_intervals_np=relev_intervals_np[sorted_indices,:]                         
				relev_features_np=relev_features_np[sorted_indices,:]
				
				relevant_entries_np[otherseq_key][key]={}
				relevant_entries_np[otherseq_key][key]["intervals"]=relev_intervals_np
				relevant_entries_np[otherseq_key][key]["features"]=relev_features_np
				
		return relevant_entries_np
