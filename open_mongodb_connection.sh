module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /home/slee232/scratch/db --port $port > mongod.log &
