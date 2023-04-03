pip install ipdb
pip install jupyter
pip install colorful
# pip install diffusers
git clone https://github.com/tumble-weed/diffusers.git
pip install -e diffusers
pip install  transformers accelerate scipy safetensors
#------------------------------------------------------------------------------------
#InSPyReNet
git clone https://github.com/plemeri/InSPyReNet
pip install -e InSPyReNet
pip install easydict
pip install timm
#------------------------------------------------------------------------------------
#lama inpainting
git clone https://github.com/tumble-weed/lama
pip install pytorch-lightning
pip install hydra-core
pip install albumentations
pip install albumentations==0.5.2
pip install webdataset
pip install pandas
pip install kornia
pip3 install wldhx.yadisk-direct
cd lama
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
cd -
#------------------------------------------------------------------------------------
# pip install rembg
pip install rembg[gpu]
# https://drive.google.com/file/d/1Tag_nzrNDCmByC2tFlEFMTN0tiYaPCQ7/view?usp=sharing
gdown 1Tag_nzrNDCmByC2tFlEFMTN0tiYaPCQ7
# download mask of generated1
gdown 1ZErvirt-KwpQjRzkPoOL44VuRZAUPrzp
gdown 1T7uG-sAukYLtHxOOei5rbBxUZRb4KaMn
gdown 1u_b7EYo-7d6iUXMw66ATtwyGLLtR7QB3
gdown 1YeGxoDdDauy0BLzYX0oJ4bs_c4-yGfBA

#.................................................................................
# simplest flask 4
gdown 16PuND-X7fOeMkcrf6VmtwFPCDHxKlgBK
# flask 3 with strap
gdown 1jqdAdSKWmtXuKsAgIHybf5xPqiIBEtKz
# flask 2 with hole
gdown 109HBAFijbxgu6rdRhAA_g42m49HdYZs3
# flask 1 with green background
gdown 1YeGxoDdDauy0BLzYX0oJ4bs_c4-yGfBA
#.................................................................................

# harmonization
git clone https://github.com/tumble-weed/Harmonizer
#.................................................................................
# clip
#download clipseg
git clone https://github.com/tumble-weed/clipseg
pip install -e clipseg
bash clipseg/setup_clip.sh

#.................................................................................
# controlnet
pip install controlnet_aux

