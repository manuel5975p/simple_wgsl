set terminal pngcairo size 1200,600 enhanced font "sans,12"
set output "bench/fft_optimal_results.png"

set title "FFT Optimal Plan — Execution Time by Size (RTX 3070)" font ",14"
set xlabel "FFT Size N"
set ylabel "Time (ms)"
set grid
set key top left

set xrange [1:260]
set yrange [0:18]

# Color by strategy using stringcolumn
plot "bench/fft_results.dat" using (stringcolumn(3) eq "CT" ? $1 : 1/0):2 with points pt 7 ps 0.8 lc rgb "#2196F3" title "Cooley-Tukey", \
     "bench/fft_results.dat" using (stringcolumn(3) eq "Blue" ? $1 : 1/0):2 with points pt 7 ps 0.8 lc rgb "#FF5722" title "Bluestein", \
     "bench/fft_results.dat" using (stringcolumn(3) eq "DFT" ? $1 : 1/0):2 with points pt 7 ps 0.8 lc rgb "#4CAF50" title "Direct DFT"
