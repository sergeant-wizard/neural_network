set terminal png
set output "residue.png"
plot "residue_momentum_decay.txt", "residue_nothing.txt"
