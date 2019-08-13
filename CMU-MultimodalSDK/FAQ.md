**Frequently Asked Questions**

1. What does the SDK do for me? 

CMU Multimodal SDK helps standardize data sharing and multimodal modeling for the field of multimodal machine learning. It provides functionalities not available in other resources. These functionalities include alignment and neural models that are not available in other libraries. 

2. SDK keeps developing, how do I keep track?

Check for the version of the SDK and make sure you are up to date. Furthermore, try to pull the code every now and then. 

3. Computational sequences keep getting more and more advanced, how do I keep up?

The goal of SDK is to make sure users always have access to the state of the art. Therefore, results may improve naturally over time, just because you have access to better tools. Check the version of the computational sequences to make sure you are using the most recent ones. If you pull the SDK and download the computational sequences from standard datasets, you will always download the most recent. 

4. How do I check my results with previous work, if computational sequences keep on improving?

It is a reasonable expectation that a researcher should dedicate some time to recreating previous works. However, with the new computational sequences, some models may improve naturally. You should run the previous works on the newest data always, if your approach uses the newest data. The implementations for most of the previous works are either available through auhtors works (ours is available in this link). Upon release of mmmodelsdk implementing most of the previous works should be relatively easy. The combination of mmmodelsdk and mmdatasdk should make running experiments very easy to make sure you are comparin to fair baselines.

5. I found an issue in the code or in a computational sequence, what do I do?

Please report it ASAP on the issues tab in github or simply contact us through email. We normally respond within 12 hours to the most urgent issues. 

6. Can I add computational sequences or add a new dataset to the SDK?

Of course, mmdatasdk is built for that purpose. We will release tutorials on how to release your own dataset. If you are familiar with the format of computational sequences, that should be relatively easy to do. Furthermore, we are also implementing hasing checks to allow researchers to register their data, and simply share the hash to allow others to recreate their experiments reliably. 

7. What about the old implementation of the SDK? 

Through the very first implementation of the SDK, we learned a lot about necessary optimizations and how to manage the datasets. At this point in time the old version of SDK is completely deprecated. Please use the new SDK, called CMU Multimodal SDK (you are in the correct gituhb). 

8. Do you share raw data?

Yes, we do here is the link: http://immortal.multicomp.cs.cmu.edu/raw_datasets/. But we don't advocate going solo on processing everything again (recreating the wheel). If you want to process your own features, see item 6 in this FAQ.

9. Do you share the data for publications prior to mid 2018?

Yes, we do here is the link: http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/. These are exact data used for our experiments, already aligned at word level. You can certainly use this data, but I do advocate exploring the datasets using the SDK. For example try different alignments, or strategies. (Please note that CMU-MOSEI had some issues for some videos over their acoustic modality. They are now solved and CMU-MOSEI downloaded from SDK gets better performance than the one we ran experiments on for original paper)

