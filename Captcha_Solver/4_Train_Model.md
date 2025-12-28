# Train Model
In a lab setting, training a model for hours is boring. We are opting to use a script to:
- Generate 10,000 CAPTCHA images locally (increased training material to improve accuracy)
- Numeric-only simple CAPTCHA for speed
- Uses a ResNet-18 or a Small CNN architecture
- Trains for only 10 epochs (should not take too long on 20 threads)
- Includes a "Validation Split" to hold back 20% of the data for testing
- Saves the result as captcha_model.pth

## Create Training Script
1. Create the training script from [train.py](train.py)
2. Run the script: `python train.py`
    - Watch Task Manager > Performance > CPU
    - You should see all 20 threads engaging as it generates and processes the images

NOTE The script reports the "loss". It's the "Error Score" or the "Difference between the AI's guess and the truth." In cybersecurity terms, it's like a lock-picking simulator. The more you practice, the fewer mistakes you make.
- Expected to go down in each succeeding epoch
- You should see a significant drop between Epoch 1 and Epoch 2, with the decrease becoming smaller (plateauing) as it reaches Epoch 10

## Learn More
Compare results/accurace with 5000 images and 5 epoch (faster less intensive training)
- How what tradeoff is there between between training data size and number of epochs?
If you switch to the XPU (using the GPU with PyTorch)
- How did this affect the training time?

The Three Variables
- Training Data - The "experience" the model gains. More data usually leads to better performance because the model sees more variations of noise and distortion.
- Epochs - How many times the model "re-reads" the entire dataset
- Training Time: The literal clock time spent processing. This is limited by your hardware (the 20 threads of the Core Ultra).

Avoiding the Traps
- Underfitting (too little data/epochs)
  - model hasn't seen enough examples or had enough practice to find a pattern
  - symptom: high loss on both training and test data
  - result: low accuracy
- Overfitting (too many epochs)
  - model starts "memorizing" the specific noise in the training images rather than learning the actual shape of the numbers
  - like a student memorizing the answers to the practice text but doesn't understand the math
- Best practices
  - Instead of stopping at an arbitrary epoch number, monitor the vlidation loss. If the validation loss stops decreasing for 2 or 3 epochs, stop training immediately. Any further training is just overfitting.
  - Data Augmentation over Raw Volume
    - If you have 10000 images, you can artifically create 50,000 by slightly rotating or blurring them in memory
    - More effecting than just more unique images because it teaches the model to be "robust" against tilt and blur  
  - Learning Rate Decay
    - Start with a higher learning rate to learn quickly, then "slow down" the learning rate as you reach the later epochs. This allows the model to fine-tune its weights without "overshooting" the optimal solution.

IMPORTANT for a AI PC with a NPU, the best practice is to train "just enough" on the CPU/GPU, then focus on Quantization. A slightly less accurate model that runs 10x faster on the NPU is often more valuable than a perfect model that is too slow to use.
