mounts : 
###
      "source": "/home/sharmistha-choudhury/sky_deploy"
      "target": "/mnt/apps"

###

setup command : 
###
          conda create -n serving python=3.12 && conda activate serving && pip install fastapi uvicorn mlflow==2.12.2 pillow numpy && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118      

###
command : 
###
         conda activate serving && python /mnt/apps/script.py --model-path $MODEL_DIR    

###
Readiness Probe Path
###
             /health
###
envs:
###
       "MODEL_DIR": "/mnt/ml"
       "ENV": "production"

###

Prediction / inference : 
###
      python predict.py --url <url> --token <serving token> --image bag.png

###
