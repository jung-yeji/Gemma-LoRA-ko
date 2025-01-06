#/bin/bash
python convert-adapter.py --input_folder "InferenceResult/backup" --output_file "preprocessing.json"

CUDA_VISIBLE_DEVICES=0 python comet-score.py --input_file "preprocessing.json" \
                 --output_csv_adapter "adapter.csv" \
                 --output_csv_gemma "gemma.csv" \
                 --batch_size 20 \
                 --use_gpu True

python convert-fft.py --input_folder "InferenceResultFFT" --output_file "preprocessing-FFT.json"

python comet-score.py --input_file "preprocessing-FFT.json" \
                 --output_csv_adapter "FFT.csv" \
                 --output_csv_gemma "FFT_gemma.csv" \
                 --batch_size 20 \
                 --use_gpu True

python matching.py --json_file preprocessing.json --csv_file adapter.csv --output_csv_file updated_adapter.csv
python matching.py --json_file preprocessing-FFT.json --csv_file FFT.csv --output_csv_file updated_FFT.csv
python matching_gemma.py --json_file preprocessing.json --csv_file gemma.csv --output_csv_file updated_gemma.csv

python comet-vis-adapter.py --csv_file_adapter updated_adapter.csv --csv_file_gemma updated_gemma.csv --output_dir output/adapter
python comet-vis-fft.py --csv_file_adapter updated_FFT.csv --csv_file_gemma updated_gemma.csv --output_dir output/FFT


python mismatch_language.py --input_file preprocessing-FFT.json --output_file mismaching/test-fft.csv --is_adpater True
python mismatch_language.py --input_file preprocessing.json --output_file mismaching/test-adapter.csv --is_adpater True
python mismatch_language.py --input_file preprocessing.json  --output_file mismaching/test-gemma.csv
