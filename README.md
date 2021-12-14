# Задание по курсу "Распределенные системы"

1 - Протокол голосования при размножении файлов. Реализовать программу, моделирующую выполнение протокола голосования для 11 файловых серверов при помощи пересылок MPI типа точка-точка. Получить временную оценку времени выполнения одним процессом 3-х операций записи и 10 операций чтения N байтов информации с файлом, расположенным (размноженным) на 11 серверах. Определить оптимальные значения кворума чтения и кворума записи для N=300. Время старта равно 100, время передачи байта равно 1 (Ts=100,Tb=1).

2 - Доработать MPI-программу, реализованную в рамках курса “Суперкомпьютеры и параллельная обработка данных”. Добавить контрольные точки для продолжения работы программы в случае сбоя.
