# Run from experiments/paper_evidence (after mkdir -p figures). Figure width matches the official COLM text width.
set terminal pdfcairo enhanced color font "Helvetica,10" size 5.5in,4.4in
set datafile separator comma
set output 'figures/incident-recovery.pdf'
set multiplot
set border 3
set tics nomirror
set grid ytics lc rgb '#dddddd'
set lmargin at screen 0.13
set rmargin at screen 0.98
set bmargin at screen 0.59
set tmargin at screen 0.92
set xlabel 'Incident update'
set ylabel 'Pass fraction'
set xrange [1:130]
set yrange [-0.03:1.05]
set key top left font ',9' opaque
set arrow 1 from 61,0 to 61,1.03 nohead dt 2 lc rgb '#777777'
set title '(a) Recorded proxy versus reference'
plot 'evidence/incident.csv' using 1:2 with lines lw 1.5 lc rgb '#D55E00' title 'Proxy pass rate', '' using 1:3 with lines lw 1.5 lc rgb '#0072B2' title 'Reference accuracy'
unset arrow 1
set bmargin at screen 0.10
set tmargin at screen 0.43
set xlabel 'Fresh recovery update'
set ylabel 'Reference batch accuracy'
set xrange [1:40]
set title '(b) Recovery from three checkpoints'
set key bottom right font ',9' opaque
plot 'evidence/recovery.csv' using (strcol(1) eq 'recovery_rec'?$3:1/0):5 with lines lw 1.5 lc rgb '#0072B2' title 'Checkpoint 60', '' using (strcol(1) eq 'recovery_reclast'?$3:1/0):5 with lines lw 1.5 lc rgb '#D55E00' title 'Checkpoint 130', '' using (strcol(1) eq 'recovery_ckpt80'?$3:1/0):5 with lines dt 2 lw 1.5 lc rgb '#009E73' title 'Checkpoint 80 (supplementary)'
unset multiplot
set terminal pdfcairo enhanced color font "Helvetica,9" size 5.5in,2.7in
set output 'figures/systems.pdf'
set multiplot layout 1,2 margins 0.10,0.98,0.22,0.89 spacing 0.17,0.05
set title '(a) Raw capture cost: 1 GPU' font ',9'
unset key
set xlabel 'Independent pair'
set ylabel 'Inclusive time change (%)'
set xrange [0.5:3.5]
set yrange [-2.3:1]
set xtics 1,1,3
set arrow 1 from graph 0,first 0 to graph 1,first 0 nohead dt 2 lc rgb '#777777'
plot 'evidence/capture_overhead_compact.csv' using 1:($4*100) with points pt 7 ps 1.2 lc rgb '#0072B2'
unset arrow 1
set title '(b) Queries: precomputed labels' font ',9'
set xlabel 'Retained trajectories'
set ylabel 'Query-stage latency (s)'
set logscale xy
set xtics ('10^4' 10000, '10^5' 100000, '10^6' 1000000)
set xrange [8000:1400000]
set yrange [0.3:70]
set key top left font ',8' spacing 1.2
plot 'evidence/query-scaling-1045762-summary.csv' using 1:($1<=41600?$4:1/0):5:6 with yerrorbars pt 7 lc rgb '#0072B2' title 'Real prefix', '' using 1:($1>41600?$4:1/0):5:6 with yerrorbars pt 5 lc rgb '#D55E00' title 'Synthetic'
unset multiplot
unset output
