

CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 1 enko > ./exper/TrainingReport/Training_Translate1-enko.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 1 koen > ./exper/TrainingReport/Training_Translate1-koen.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 3 cnko > ./exper/TrainingReport/Training_Translate3-cnko.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 3 kocn > ./exper/TrainingReport/Training_Translate3-kocn.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 3 jpko > ./exper/TrainingReport/Training_Translate3-jpko.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 200 8 3 kojp > ./exper/TrainingReport/Training_Translate3-kojp.Rank8.200.txt


CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 1 enko > ./exper/TrainingReport/Training_Translate1-enko.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 1 koen > ./exper/TrainingReport/Training_Translate1-koen.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 3 cnko > ./exper/TrainingReport/Training_Translate3-cnko.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 3 kocn > ./exper/TrainingReport/Training_Translate3-kocn.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 3 jpko > ./exper/TrainingReport/Training_Translate3-jpko.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 400 8 3 kojp > ./exper/TrainingReport/Training_Translate3-kojp.Rank8.400.txt


CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 1 enko > ./exper/TrainingReport/Training_Translate1-enko.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 1 koen > ./exper/TrainingReport/Training_Translate1-koen.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 3 cnko > ./exper/TrainingReport/Training_Translate3-cnko.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 3 kocn > ./exper/TrainingReport/Training_Translate3-kocn.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 3 jpko > ./exper/TrainingReport/Training_Translate3-jpko.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 800 8 3 kojp > ./exper/TrainingReport/Training_Translate3-kojp.Rank8.800.txt


CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 1 enko > ./exper/TrainingReport/Training_Translate1-enko.Rank8.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 1 koen > ./exper/TrainingReport/Training_Translate1-koen.Rank8.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 3 cnko > ./exper/TrainingReport/Training_Translate3-cnko.Rank8.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 3 kocn > ./exper/TrainingReport/Training_Translate3-kocn.Rank8.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 3 jpko > ./exper/TrainingReport/Training_Translate3-jpko.Rank8.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_Translate.py 1600 8 3 kojp > ./exper/TrainingReport/Training_Translate3-kojp.Rank8.1600.txt


CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_QA.py 200 8  > ./exper/TrainingReport/Training_QA.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_QA.py 400 8  > ./exper/TrainingReport/Training_QA.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_QA.py 800 8  > ./exper/TrainingReport/Training_QA.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_QA.py 1600 8  > ./exper/TrainingReport/Training_QA.Rank8.1600.txt

CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryConver.py 200 8  > ./exper/TrainingReport/Training_SummaryConv.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryConver.py 400 8  > ./exper/TrainingReport/Training_SummaryConv.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryConver.py 800 8  > ./exper/TrainingReport/Training_SummaryConv.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryConver.py 1600 8  > ./exper/TrainingReport/Training_SummaryConv.Rank8.1600.txt

CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryLAW.py 200 8  > ./exper/TrainingReport/Training_SummaryLAW.Rank8.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryLAW.py 400 8  > ./exper/TrainingReport/Training_SummaryLAW.Rank8.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryLAW.py 800 8  > ./exper/TrainingReport/Training_SummaryLAW.Rank8.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeLoRA_SummaryLAW.py 1600 8  > ./exper/TrainingReport/Training_SummaryLAW.Rank8.1600.txt





CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 1 enko > ./exper/FFTTrainingReport/FFTTraining_Translate1-enko.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 1 koen > ./exper/FFTTrainingReport/FFTTraining_Translate1-koen.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 3 cnko > ./exper/FFTTrainingReport/FFTTraining_Translate3-cnko.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 3 kocn > ./exper/FFTTrainingReport/FFTTraining_Translate3-kocn.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 3 jpko > ./exper/FFTTrainingReport/FFTTraining_Translate3-jpko.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 200 3 kojp > ./exper/FFTTrainingReport/FFTTraining_Translate3-kojp.200.txt


CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 1 enko > ./exper/FFTTrainingReport/FFTTraining_Translate1-enko.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 1 koen > ./exper/FFTTrainingReport/FFTTraining_Translate1-koen.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 3 cnko > ./exper/FFTTrainingReport/FFTTraining_Translate3-cnko.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 3 kocn > ./exper/FFTTrainingReport/FFTTraining_Translate3-kocn.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 3 jpko > ./exper/FFTTrainingReport/FFTTraining_Translate3-jpko.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 400 3 kojp > ./exper/FFTTrainingReport/FFTTraining_Translate3-kojp.400.txt


CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 1 enko > ./exper/FFTTrainingReport/FFTTraining_Translate1-enko.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 1 koen > ./exper/FFTTrainingReport/FFTTraining_Translate1-koen.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 3 cnko > ./exper/FFTTrainingReport/FFTTraining_Translate3-cnko.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 3 kocn > ./exper/FFTTrainingReport/FFTTraining_Translate3-kocn.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 3 jpko > ./exper/FFTTrainingReport/FFTTraining_Translate3-jpko.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 800 3 kojp > ./exper/FFTTrainingReport/FFTTraining_Translate3-kojp.800.txt


CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 1 enko > ./exper/FFTTrainingReport/FFTTraining_Translate1-enko.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 1 koen > ./exper/FFTTrainingReport/FFTTraining_Translate1-koen.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 3 cnko > ./exper/FFTTrainingReport/FFTTraining_Translate3-cnko.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 3 kocn > ./exper/FFTTrainingReport/FFTTraining_Translate3-kocn.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 3 jpko > ./exper/FFTTrainingReport/FFTTraining_Translate3-jpko.1600.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_Translate.py 1600 3 kojp > ./exper/FFTTrainingReport/FFTTraining_Translate3-kojp.1600.txt


CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_QA.py 200  > ./exper/FFTTrainingReport/FFTTraining_QA.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_QA.py 400  > ./exper/FFTTrainingReport/FFTTraining_QA.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_QA.py 800  > ./exper/FFTTrainingReport/FFTTraining_QA.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_QA.py 1600  > ./exper/FFTTrainingReport/FFTTraining_QA.1600.txt

CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryConver.py 200  > ./exper/FFTTrainingReport/FFTTraining_SummaryConv.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryConver.py 400  > ./exper/FFTTrainingReport/FFTTraining_SummaryConv.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryConver.py 800  > ./exper/FFTTrainingReport/FFTTraining_SummaryConv.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryConver.py 1600  > ./exper/FFTTrainingReport/FFTTraining_SummaryConv.1600.txt

CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryLAW.py 200  > ./exper/FFTTrainingReport/FFTTraining_SummaryLAW.200.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryLAW.py 400  > ./exper/FFTTrainingReport/FFTTraining_SummaryLAW.400.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryLAW.py 800  > ./exper/FFTTrainingReport/FFTTraining_SummaryLAW.800.txt
CUDA_VISIBLE_DEVICES=0 python3 makeFullFT_SummaryLAW.py 1600  > ./exper/FFTTrainingReport/FFTTraining_SummaryLAW.1600.txt
