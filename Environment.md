# PricePredict Working Environment

# Anaconda Python 3.12.6

# Keras & Scikit-learn

# Pandas

# Jupyter Notebook

# TA-Lib

## [TA-Lib Installation](https://mrjbq7.github.io/ta-lib)

- 150+ Indicators and Candlestick Pattern Recognition

```shell
cd workspace
# Get TA-Lib "C" source code
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# Extrace the code into the workspace directory
tar -xzf ta-lib-0.4.0-src.tar.gz
# Change to the extracted directory
cd ta-lib/
# Configure the build
./configure --prefix=/usr/share
# Build the library
make
# Install the library
sudo make install
# Create symbolic links so that ta-lib can be found by the pip installer
cd /usr/share/lib
sudo ln -s libta_lib.a libta-lib.a
sudo ln -s libta_lib.la libta-lib.la
sudo ln -s libta_lib.so libta-lib.so
sudo ln -s libta_lib.so.0 libta-lib.so.0
sudo ln -s libta_lib.so.0.0.0 libta-lib.so.0.0.0
# Change back to the pricepredict workspace directory
cd ~/workspace/pricepredict/lib/
# Invokde the conda pricepredict environment
conda activate pricepredict
# Install the ta-lib python wrapper
pip install ta-lib
```

# My Environment [@dsidlo]

- Computer
  - Intel(R) Xeon(R) W-11955M CPU @ 2.60GHz
    - 8 Hyper-Threading Cores
  - RAM 128GB
  - A5000 GPU (16GB)
- Ubuntu  22.04.5 LTS
- PyCharm 2024
- Anaconda Python 3.12.6
