curl -O https://archive.ics.uci.edu/static/public/280/higgs.zip
tar -xf higgs.zip
gunzip HIGGS.csv.gz
python3 converter.py
rm higgs.zip HIGGS.csv
