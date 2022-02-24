for f in *_ref.brep
do
    base=${f%_ref.brep}
    new_f=${base}_tst.brep
    #echo -f $new_f $f
    cp -f $new_f $f
done
