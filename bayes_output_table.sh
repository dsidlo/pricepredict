#!/bin/bash

echo "Paste log file here:"

cat > bayes_tests.txt

cat bayes_tests.txt | perl -ne 'if ($_ =~ m/^\|/) {print "$_";}' > bayes_output_table.csv
