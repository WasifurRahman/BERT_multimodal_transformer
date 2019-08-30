module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /scratch/slee232/db --port $port > mongod.log &
