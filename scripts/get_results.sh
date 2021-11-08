chmod +x scripts/gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1HOqS6f2fd6BcAwESOQJPCrDTl-J-SwVD /tmp/results.zip
unzip /tmp/results.zip -d /tmp
mv /tmp/results results
