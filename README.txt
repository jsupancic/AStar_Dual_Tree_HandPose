# AStar_Dual_Tree_HandPose
A fast NN hand-pose estimation system using dual-KD-trees.
Also includes implementations for various-SVM (inc. part models) and
tree based methods.

Please direct questions to jsupanci@uci.edu

Compile w/ 
(1) cmake . 
(2) make -jX

Run as
./deformable_depth eval_model CFG=myconfig_file.cfg

# Configuration file options:
HEAP_PROF=FALSE # profile where the memory is being used...
JOINT_C=10 # for using SVMs
NUM_CPUS=20 # how many CPUs to use?
MAX_TRAIN_MINS=150 # cut off training at this point.
OBJ_DEPTH=20 # how deep is the bounding box
SKIN_FILTER=TRUE # use a high recall skin detection pre-processor
MANIFOLD_FN=ApxMin # which depths to consider?
NEG_SET=BIG # how many negatives to use when training? 
TEST_VIDEO=TRUE # test on the video sequences? 
# Cheat options for debugging
CHEAT_DEPTH=FALSE # give ground truth depth
CHEAT_HAND_BB=FALSE # give ground truth bounding box
# .985 is the highest we can go...
CHEAT_HAND_BB_THRESH=.8
CHEAT_HAND_ORI=FALSE # give ground truth orientation
#CHEAT_HAND_LR=RIGHT
CHEAT_ONLY_FREESPACE=FALSE
# DONT_USE_ARM=TRUE
OUT_DIR=out/
#OUT_DIR=automatic
# ApxNN, KinectPose, AONN, DeepYi, FLANN, NYUModel, Human, VolNN, Export, etc.
# Keskin
MODEL=ApxNN # use fast NN
WRITE_MODEL=FALSE
# SAVED_MODEL=/home/jsupanci/Dropbox/out/2014.09.01-ICL-NN25k20-SavedTree/model.yml
#SORT_NODES=
SORT_TO_DEPTH=1
ORTHOGRAPHIC=FALSE # tends to introduce artifacts which hurts performance.
# for training decision trees
TREE_SPLIT_SIZE=50
ENTROPY_TYPE=shannon

# Configure FLANN
FLANN_SAMPLES=50
FLANN_LSH_TABLE_NUMBER=15
FLANN_LSH_KEY_SIZE=15
FLANN_LSH_MULTI_PROBE_LEVEL=2

# configure the dataset
#DATASET=NYU
#DATASET=synth
DATASET=ICL
#DATASET=DIRECTORIES
#DATASET=EGOCENTRIC_SYNTH
#DATASET=test_videos
#DIRECTORIES=/home/grogez/Egocentric_Synth_Poser/
#DATASET=NYU
ICL_BASE=/home/jsupanci/workspace/data/ICL_HANDS2/
#ICL_BASE=/extra/titansc0/jsupanci/data/ICL_HANDS2/

#SYNTHETIC_DIRECTORY=/mnt/data/jsupanci/Synth-75000-WristVariation/
#SYNTHETIC_DIRECTORY=/home/jsupanci/data/synth-yi/
SYNTHETIC_DIRECTORY=/mnt/big/shared_data/Synth-75000-WristVariation/
#SYNTHETIC_DIRECTORY=/extra/titansc0/jsupanci/data/Synth-75000-WristVariation/
LEVEL_DB=/scratch/jsupanci/leveldb/
#LEVEL_DB=data/annotation-database

#DATASET=test_videos
#SYNTHETIC_DIRECTORY=./data/oracle_synth_armless_depth/
# setup the 500 exemplar model
# 3000  / 500
# 15000 / 5000
#NPOS=15000
#NPOS=25000
NPOS=2500
NNEG=0
#SYNTHETIC_DIRECTORY=data/2014.05.14-Synth50
#SYNTHETIC_DIRECTORY=/mnt/big/shared_data/Synth-75000-WristVariation/
#SYNTHETIC_DIRECTORY=/mnt/big/shared_data/Synth-75000-WristVariation/
IMPLICIT_IN_PLANE_ROTATION=TRUE
DATA_SOURCE_PRE_ROTATE=TRUE
TRAINING_INSERT_LR_FLIPS=FALSE
# kmeans, pyramid
ASTAR_ADD_BOUNDING=FALSE
ASTAR_CLUSTERING_ALG=kmeans
DEBUG_ASTAR_VIDEO=FALSE
NN_XTEMPL_CACHE_SIZE=50000
# no pruning with 5000000, 500000
NN_SMA_OPENED_LIMIT=50000
NN_ADMISSIBILITY=1
NN_PYRAMID_SHARPNESS=.5
SEGMENTATION=FALSE
NN_LOG_TREE=FALSE

# Analysis Section - universal parameters
POST_POSE_CONVEXITY=FALSE 
POST_POSE_VORONOI=TRUE    # this one helps
INTERPOLATE_TRACK=FALSE # tries to make tracker evaluation more fair.
INTERPOLATE_PARTS_IK=FALSE
# factor by 2.5 to 4.
FINGER_DIST_THRESH=4
SCORE_SKIP_LEFT=FALSE
# Right dominstates in egocenric
SCORE_SKIP_RIGHT=FALSE

# Synthesis parameters
SYNTH_CLUSTERS=5
SYNTH_EX_PER_CLUSTER=25000
SYNTH_PERTURB_R_MIN=20
SYNTH_PERTURB_R_MAX=120
SYNTH_FINGER_AREA_FILTER=FALSE
