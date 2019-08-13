#word_level_align.py
#first aligns a dataset to the words vectors and collapses other modalities (by taking average of them for the duration of the word). After this operation every modality will have the same frequency (same as word vectors). Then the code aligns based on opinion labels (note that collapse does not happen for this step.

import mmsdk
from mmsdk import mmdatasdk
import numpy

#uncomment all the ==> lines together

#A simple averaging technique. More advanced methods can be built based on intervals.
def myavg(intervals,features):
        return numpy.average(features,axis=0)


#Downloading the dataset
cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')



#some random video from cmumosi_highlevel
#==>some_video=list(cmumosi_highlevel["glove_vectors"].data.keys())[0]


#Aligning to the words to get word-level alignments
cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])



#get the intervals and features accompanying the 100th word in the some_video
#==>some_video_100th_word=some_video+'[100]'
#==>for compseq_name in list(cmumosi_highlevel.computational_sequences.keys()):
#==>	compseq=cmumosi_highlevel[compseq_name]
#==>	print (compseq_name)
#==>	print (numpy.array(compseq.data[some_video_100th_word]["intervals"]).shape,numpy.array(compseq.data[some_video_100th_word]["features"]).shape)
#==>	print ("-------")


#Aligning to the computational labels, thus removing the unsupervised components of CMU-MOSI

cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
cmumosi_highlevel.align('Opinion Segment Labels')



#get the intervals and features accompanying the 2nd in some_video
#==>some_video_2nd_segment=some_video+'[2]'
#==>for compseq_name in list(cmumosi_highlevel.computational_sequences.keys()):
#==>	compseq=cmumosi_highlevel[compseq_name]
#==>	print (compseq_name)
#==>	print (numpy.array(compseq.data[some_video_2nd_segment]["intervals"]).shape,numpy.array(compseq.data[some_video_2nd_segment]["features"]).shape)
#==>	print ("-------")

#Deploying the files to the disk and reading them again - Building machine learning models start right after this. No need to do alignment multiple times since aligned files can be deployed and used again.
deploy_files={x:x for x in cmumosi_highlevel.computational_sequences.keys()}
cmumosi_highlevel.deploy("./deployed",deploy_files)
aligned_cmumosi_highlevel=mmdatasdk.mmdataset('./deployed')



