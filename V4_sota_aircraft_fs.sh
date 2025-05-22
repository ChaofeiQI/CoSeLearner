#############################
# CoSeLearner & CoSeDistiller
#############################
export GPU="3"

##############
# Query=15
##############
# 5way-1shot
python3 main1_meta.py     --config config/query_15/5way_1shot_aircraft-fs.py  --device $GPU --mode train  --log_step 5
python3 main1_meta.py     --config config/query_15/5way_1shot_aircraft-fs.py  --device $GPU --mode eval

# 5way-1shot-distill-3
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 3 --device $GPU --mode train --log_step 5
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 3 --device $GPU --mode eval
# 5way-1shot-distill-4
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 4 --device $GPU --mode train --log_step 5
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 4 --device $GPU --mode eval
# 5way-1shot-distill-5
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 5 --device $GPU --mode train --log_step 5
python3 main2_distill.py  --config config/query_15/5way_1shot_aircraft-fs.py  --gen_stu 5 --device $GPU --mode eval
