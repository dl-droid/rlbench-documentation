echo "Downloading V-Rep/CoppeliaSim"
wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz --no-check-certificate 
tar -xvf CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz
rm CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz
echo "V-Rep/CoppeliaSim Downloaded. Installing Drivers"
sudo apt-get update && sudo apt-get install -y \
  zsh \
  wget \
  libglib2.0-0  \
  libgl1-mesa-glx \
  xcb \
  "^libxcb.*" \
  libx11-xcb-dev \
  libglu1-mesa-dev \
  libxrender-dev \
  libxi6 \
  libdbus-1-3 \
  libfontconfig1 \
  xvfb \
  tar \
  git \
  qtbase5-dev qtdeclarative5-dev libqt5webkit5-dev libsqlite3-dev qt5-default qttools5-dev-tools \
  && sudo rm -rf /var/lib/apt/lists/*
echo "Drivers Downloaded. Installing RLBench"
git clone https://github.com/stepjam/PyRep.git
echo "Setting Environment Variables For A Test Run"
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
git clone https://github.com/valayDave/RLBench.git
echo "Installing Virtual Environment"
python3 -m venv .env
echo "Installing PyRep and RLBench in Virtual Environment"
.env/bin/pip install -r PyRep/requirements.txt
.env/bin/pip install -e PyRep
.env/bin/pip install -r RLBench/requirements.txt
.env/bin/pip install -e RLBench
# echo "Running Demo Script"
# .env/bin/python PyRep/examples/example_baxter_pick_and_pass.py
echo "Installed PyRep and RLBench !"
echo "Installing PyTorch and Metaflow"
.env/bin/pip install -r requirements.txt
# docker run -i -t --rm -e DISPLAY=$DISPLAY -u docker -v /tmp/.X11-unix:/tmp/.X11-unix:ro --name="PyRep_Docker_Test" valaygaurang/pyrep:0.1
echo "Downloading Dataset with 200 Demos Of Reach Target Task"
.env/bin/python RLBench/tools/dataset_generator.py --tasks reach_target --episodes_per_task 200 --processes 1