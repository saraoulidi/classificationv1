#!/bin/sh

# Warnning this script should be executed from inside the scripts dir.!!!

clear

# Dependencies check
if command -v mogrify > /dev/null 2>&1
then
    echo "mogrify found"
    echo "$(mogrify -version | head -1)"
else
    echo "mogrify not found!! Make sure you have Image mogrify installed on your machine."
    echo "if you are on GNU/Linux (Debian based distros): apt install imagemagick"
    echo "if you are on Mac: brew install imagemagick"
    echo "if you are on Windows good luck."
    exit 1;
fi

if command -v fdupes > /dev/null 2>&1
then
    echo "fdupes found"
    echo "Version: $(fdupes -v)"
else
    echo "fdupes not found!! Make sure you have fdupes installed on your machine."
    echo "if you are on GNU/Linux (Debian based distros): apt install fdupes"
    echo "if you are on Mac: brew install fdupes"
    echo "if you are on Windows good luck."
    exit 1;
fi

# This path should not be hard coded into the script
# Not everyone will have the same path
# cd ~/Desktop/Projects/downloads/pwc

# $1
datasetsBaseDir="../Scraper/download" # without the trailing slash plz

randomUUID=$(uuidgen)

# 1st thing 1st let's download all the images from the links we already have.
# This for loop will loop through all .txt files in the urls dir.
for urlsFile in ../Scraper/urls/*.txt
do
    # now let's download all the urls from the current file into a new dedicated folder.
    # 1st let's generate a uniq name for the dir.
    brandName=$(echo $urlsFile | cut -d'/' -f4) # split the snake (I mean the file path :p) and get the last chunk.
    brandName=$(echo $brandName | sed "s/.txt//g") # let's replace .txt with nothing
    brandName=$(echo $brandName | sed "s/urls-//g") # let's replace urls- with nothing
    brandName=$(echo $brandName | egrep  -o '[a-z]+')

    echo "$(date) Doing $brandName!!";

    # Dataset Download Directory
    ddd="$datasetsBaseDir/$randomUUID/$brandName"

    # @TODO: Zakarya & Sara
    # download images into the new dir using the brand name & the random uuid
    cat $urlsFile | xargs -n 1 -P 20 wget -q --no-check-certificate --tries=1 --timeout=5 --directory-prefix=$ddd

    totalDownlaod=$(ls -1a $ddd| wc -l)

    # remove duplicated files
    nbrOfDuplicates=$(fdupes -r -f $ddd | grep -v '^$' | wc -l)
    # delete
    fdupes -r -f $ddd | grep -v '^$' | xargs rm -v

    mkdir -pv "$ddd/clean"

    # convert everything to jpg in one go.
    mogrify -path $ddd/clean/ -format jpg "$ddd/*"

    mkdir -pv $ddd/clean

    cp -v $ddd/*.jpg $ddd/clean/

    # convert all .jpeg to jpg
    convert "$ddd/*.jpeg" "$ddd/clean/*.jpg"

    # convert all .png to jpg
    convert "$ddd/*.png" "$ddd/clean/*.jpg"

    # convert all .svg to jpg
    convert "$ddd/*.svg" "$ddd/clean/*.jpg"

    # convert all .webp to jpg
    convert "$ddd/*.webp" "$ddd/clean/*.jpg"

    counter=1
    for downloadedImage in $ddd/clean/*.*
    do
        # generate a a sequential name using a 9 digit long number.
        sequentialNameWithPadding=$(printf "%09d-digi.jpg" "$counter")
        mv -n -v "$downloadedImage" "$ddd/clean/$sequentialNameWithPadding"
        counter=$((counter+1))
    done

    python3 ./groupimg.py -f $ddd -k 12

    # print a report file at the end.
    echo "$(date) | Brand $brandName - Downloaded $totalDownlaod image(s), found $nbrOfDuplicates duplicate(s), after cleanup we have $(ls -1 $ddd/clean | wc -l) image(s)." >> $datasetsBaseDir/$randomUUID/report.txt

done

exit 0;
