cd "$(dirname "$0")"
rm -f ./out/cpu/*

cd ./stu_upload && python ./evaluate_cpu.py
