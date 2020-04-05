echo "Downloading V-Rep/CoppeliaSim"
wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz --no-check-certificate 
tar -xvf CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz
rm CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz
echo "V-Rep/CoppeliaSim Downloaded. Downloading PyRep"
git clone https://github.com/stepjam/PyRep.git
git clone https://github.com/stepjam/RLBench.git
echo "Installing Virtual Environment"
python3 -m venv .env
echo "Installing PyRep and RLBench in Virtual Environment"
.env/bin/pip install -r PyRep/requirements.txt
.env/bin/pip install -e PyRep
.env/bin/pip install -r RLBench/requirements.txt
.env/bin/pip install -e RLBench
echo "Setting Environment Variables For A Test Run"
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# echo "Running Demo Script"
# .env/bin/python PyRep/examples/example_baxter_pick_and_pass.py
echo "Installed PyRep and RLBench !"
# docker run -i -t --rm -e DISPLAY=$DISPLAY -u docker -v /tmp/.X11-unix:/tmp/.X11-unix:ro --name="PyRep_Docker_Test" valaygaurang/pyrep:0.1
