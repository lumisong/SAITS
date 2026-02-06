# Create RawData directory and navigate into it
New-Item -ItemType Directory -Force -Path RawData
Set-Location -Path RawData

# for PhysioNet-2012
New-Item -ItemType Directory -Force -Path Physio2012_mega
Set-Location -Path Physio2012_mega
Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download -OutFile set-a.tar.gz
Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download -OutFile set-b.tar.gz
Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download -OutFile set-c.tar.gz

Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download -OutFile Outcomes-a.txt
Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download -OutFile Outcomes-b.txt
Invoke-WebRequest -Uri https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download -OutFile Outcomes-c.txt

# PowerShell doesn't have a built-in tar command, so this will use a windows .exe
tar -xf set-a.tar.gz
tar -xf set-b.tar.gz
tar -xf set-c.tar.gz

New-Item -ItemType Directory -Force -Path mega
Move-Item -Path .\set-a\* -Destination .\mega
Move-Item -Path .\set-b\* -Destination .\mega
Move-Item -Path .\set-c\* -Destination .\mega

Set-Location -Path ..

# for Air-Quality
New-Item -ItemType Directory -Force -Path AirQuality
Set-Location -Path AirQuality
Invoke-WebRequest -Uri http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip -OutFile PRSA2017_Data_20130301-20170228.zip
Expand-Archive -Path .\PRSA2017_Data_20130301-20170228.zip -DestinationPath .
Set-Location -Path ..

# for Electricity
New-Item -ItemType Directory -Force -Path Electricity
Set-Location -Path Electricity
Invoke-WebRequest -Uri https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip -OutFile LD2011_2014.txt.zip
Expand-Archive -Path .\LD2011_2014.txt.zip -DestinationPath .
Set-Location -Path ..

# for Electricity Transformer Temperature (ETT)
New-Item -ItemType Directory -Force -Path ETT
Set-Location -Path ETT
Invoke-WebRequest -Uri https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv -OutFile ETTm1.csv
