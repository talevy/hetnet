for i in `seq 1 4 800`
do
    python targeted-attack.py attack-diff-data2/urchin.random.attack.difference$i.txt &
    python targeted-attack.py attack-diff-data2/urchin.random.attack.difference`expr $i + 1`.txt &
    python targeted-attack.py attack-diff-data2/urchin.random.attack.difference`expr $i + 2`.txt
done
