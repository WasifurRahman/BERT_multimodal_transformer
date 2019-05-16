module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /scratch/mhasan8/mongodb_storage/bert_multimodal_transformer --port $port > mongod.log &
