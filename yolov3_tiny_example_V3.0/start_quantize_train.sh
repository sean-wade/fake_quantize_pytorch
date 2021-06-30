python quantize_train.py \
--weights weights/voc_tiny_fp32_0.572.pt \
--cfg ./models/yolov3-tiny.yaml \
--data ./data/voc.yaml \
--batch-size 16 \
--device 3 \
--name q_mmTact_mmTwt_concatfix_16bit_IP2_adjust1417