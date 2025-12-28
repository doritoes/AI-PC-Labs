# Train Model
In a lab setting, training a model for hours is boring. We are opting to use a script to:
- Generate 5,000 CAPTCHA images locally
- Numeric-only simple CAPTCHA for speed
- Uses a ResNet-18 or a Small CNN architecture
- Trains for only 5-10 epochs (should take ~10 minutes on those 20 threads)
- Includes a "Validation Split" to hold back 20% of the data for testing
- Saves the result as captcha_model.pth

## Create Training Script
1. Create the training script from [train.py](train.py)
2. Run the script: `python train.py`
    - Watch Task Manager > Performance > CPU
    - You should see all 20 threads engaging as it generates and processes the images

NOTE The script reports the "loss". It's the "Error Score" or the "Difference between the AI's guess and the truth." In cybersecurity terms, it's like a lock-picking simulator. The more you practice, the fewer mistakes you make.
- Expected to go down in each succeeding epoch
- You should see a significant drop between Epoch 1 and Epoch 2, with the decrease becoming smaller (plateauing) as it reaches Epoch 5

## Save Model
export the trained weights as .pth or .h5 file



## Learn More
Use GPU and use more images and epochs
