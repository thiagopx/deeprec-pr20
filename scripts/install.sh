# installing python 3.5 from source on Ubuntu 18.04: https://vlearningit.wordpress.com/install-python3-5-from-the-source-in-ubuntu-18-04/
PROJECT=deeprec-pr20
BASE_DIR=$HOME
PROJECT_DIR=`pwd`
ENV_DIR=$BASE_DIR/envs/$PROJECT # directory for the virtual environemnt
QSOPTDIR=~/qsopt
CONCORDEDIR=~/concorde
PYTHON_VERSION=6
ORANGE='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

echo -e "${ORANGE}1) Preparing environment${NC}"
mkdir -p $ENV_DIR
sudo apt update
sudo apt install python3.$PYTHON_VERSION-dev python3.$PYTHON_VERSION-tk python3-pip curl -y
sudo pip3 install -U virtualenv
virtualenv -p python3.$PYTHON_VERSION $BASE_DIR/envs/$PROJECT

echo -e "${ORANGE}2) Installing Concorde${NC}"

echo -e "${BLUE}=> download${NC}"
mkdir -p $QSOPTDIR
curl -o $QSOPTDIR/qsopt.a http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a
curl -o $QSOPTDIR/qsopt.h http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h
curl -o $BASE_DIR/concorde.tgz http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar -xf $BASE_DIR/concorde.tgz -C $BASE_DIR
rm -rf $BASE_DIR/concorde.tgz

echo -e "${BLUE}=> configuration${NC}"
cd $CONCORDEDIR
./configure --with-qsopt=$QSOPTDIR

echo -e "${BLUE}=> compilation${NC}"
make

echo -e "${BLUE}=> adjusting PATH${NC}"
if ! grep -q "$CONCORDEDIR/TSP" $ENV_DIR/bin/activate ; then
   echo export PATH=\$PATH:$CONCORDEDIR/TSP >> $ENV_DIR/bin/activate
fi

echo -e "${ORANGE} 3) Installing OCR${NC}"
sudo apt install enchant -y
sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev -y

echo -e "${ORANGE} 4) Installing Python requirements${NC}"
. $ENV_DIR/bin/activate
cd $PROJECT_DIR
pip install -r requirements.txt