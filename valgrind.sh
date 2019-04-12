# https://www.cprogramming.com/debugging/valgrind.html
valgrind --tool=memcheck --leak-check=yes --log-file="leaks.txt" ./ransac 