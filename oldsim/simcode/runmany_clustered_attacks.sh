for i in `seq 1 4 800`
do
    python targeted-cluster-attack.py attack-diff-data/urchin.cluster.attack.difference$i.txt &
    python targeted-cluster-attack.py attack-diff-data/urchin.cluster.attack.difference`expr $i + 1`.txt &
    python targeted-cluster-attack.py attack-diff-data/urchin.cluster.attack.difference`expr $i + 2`.txt
done
