# Process the raw and unsorted tabular data ingested from google storage
# stitch data together and sort
# because the data is 180GB, unix sort is by far more efficient
# than sorting in python


# copies from google storage
# requires set: https://cloud.google.com/storage/docs/gsutil
gsutil cp gs://cathleen_research/* ~/khan_data/.

# concatenate the files into one file
cat ~/khan_data/math_sessions_* > ~/khan_data/khan_data_unsorted.csv

# remove duplicated headers from the new file
grep 'sha_id' ~/khan_data/khan_data_unsorted.csv | uniq > ~/khan_data/data_head
grep -v 'sha_id' ~/khan_data/khan_data_unsorted.csv| uniq > ~/sorted_data/khan_data_unsorted_nohead.csv

#  use unix sort to the data
# because the file is so large, the /tmp file runs out of space
# so create a separate directory for tmp sort files to deal with this
sort --parallel=10 --field-separator="," -k 1,1 -k 2,2n -T ~/khan_data/sort_tmp \
    ~/sorted_data/khan_data_unsorted_nohead.csv > ~/sorted_data/khan_data_sorted_nohead.csv


# add header back to sorted data
cat ~/sorted_data/data_head ~/sorted_data/khan_data_sorted_nohead.csv > \
    ~/sorted_data/khan_data_sorted.csv
rm  ~/sorted_data/khan_data_sorted_nohead.csv
