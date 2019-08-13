raw={}
raw["CMU_MOSI_ModifiedTimestampedWords"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWords.csd'
raw["phonemes"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedPhones.csd'

highlevel={}
#BERT and glove will be back soon with the newest version of forced alignment - you can extract yours 
#but make sure BERT is extracted per segment and not per video. Otherwise your method will have more information 
#beyond just segment. 
#highlevel["BERT embeddings"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedBERT.csd'
#highlevel["glove_vectors"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWordVectors.csd'
highlevel["FACET 4.1"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/visual/CMU_MOSI_VisualFacet_4.1.csd'
highlevel["OpenSmile-emobase2010"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_OpenSmile_EB10.csd'
highlevel["OpenSMILE"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_openSMILE_IS09.csd'
highlevel["OpenFace"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/visual/CMU_MOSI_OpenFace2.csd'
highlevel["COVAREP"]='http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_COVAREP.csd'
#covarep to be added
labels={}
labels["Opinion Segment Labels"]="http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd"


