setup command : 
###
          conda create -n serving python=3.12 && conda activate serving && pip install fastapi uvicorn mlflow==2.12.2 pillow numpy && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118      

###
command : 
###
         conda activate serving && python /mnt/apps/trial.py --model-path $MODEL_DIR    

###
envs:
###
       "MODEL_DIR": "/mnt/ml"
       "ENV": "production"

###
