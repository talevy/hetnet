for i in `seq 100 4 800`
do
    python alleleR.py ratiochangedata/urchin$i.txt &
    python alleleR.py ratiochangedata/urchin`expr $i + 1`.txt &
    python alleleR.py ratiochangedata/urchin`expr $i + 2`.txt &
    python alleleR.py ratiochangedata/urchin`expr $i + 3`.txt
done
