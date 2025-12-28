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



## Save Model
export the trained weights as .pth or .h5 file



## Learn More
Use GPU and use more images and epochs
