chmod +x scripts/gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1nDSQXbOcX_sUmQSmmcYVc0Ggud-ZMFiI /tmp/datasets.zip
unzip /tmp/datasets.zip -d /tmp
mv /tmp/datasets/ datasets