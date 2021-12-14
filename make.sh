#!/bin/bash -x
sudo rm -rf build/
mkdir build
cd build
cmake .. && make 
touch test_file.txt
echo "Тестовый файл для задания: Протокол голосования при размножении файлов." > test_file.txt
