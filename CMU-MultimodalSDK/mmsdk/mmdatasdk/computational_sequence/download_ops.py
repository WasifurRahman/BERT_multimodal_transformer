import h5py
import time
import requests
from tqdm import tqdm 
import os
import math
import sys
from mmsdk.mmdatasdk import log



def readURL(url,destination):
	#TODO: replace the split of destination with cross-os compatible operation
	if os.path.isdir(destination.rsplit('/',1)[-2]) is False:
		os.mkdir(destination.rsplit('/',1)[-2])
	if destination is None:
		log.error("Destination is not specified when downloading data",error=True)
	if(os.path.isfile(destination)):
		log.error("%s file already exists ..."%destination,error=True)
	r = requests.get(url, stream=True)
	if r.status_code != 200:
		log.error('URL: %s does not exist'%url,error=True) 
	# Total size in bytes.
	total_size = int(r.headers.get('content-length', 0)); 
	block_size = 1024
	wrote = 0 
	with open(destination, 'wb') as f:
		log.status("Downloading from %s to %s..."%(url,destination))
		for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='MB', unit_scale=True,leave=False):
			wrote = wrote  + len(data)
			f.write(data)
	f.close()
	if total_size != 0 and wrote != total_size:
		log.error("Error downloading the data ...")
	log.success("Download complete!")
	return True
