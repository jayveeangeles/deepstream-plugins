#!/bin/bash

echo "use $1 as working directory"
echo "use $2/lib as destination directory"
LIBSDIR=$1/../lib
SOURCE_FILES=$(find $LIBSDIR/ -type f \( -iname \*.cpp -o -iname \*.h -o -iname \*.cu \))

for i in $SOURCE_FILES
do
    h=$(basename $i .h)
    if [ $h != "ds_image" ]
        then
            echo "linking $i to $2/lib"
            ln -s $i $2/lib
    else
        echo "not doing anything with $i"
    fi
done