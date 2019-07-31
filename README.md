# Captcha Recognition
This project hosts the python code which it trains the captcha resolver and provides the output .dat and .hdf5 files

Steps to train the captcha model

1. Clone the project to your local folder.

2. Unzip the generated_captcha_images.zip in the same folder.

3. Create a new folder by name extracted_letter_images in the folder where train_model.py is present.

4. Before running the requirements.txt have pyhton 3 installed in your system.

5. From the command prompt navigate to folder where extract_single_letters_from_captchas.py is present and then run "python extract_single_letters_from_captchas.py".

6. Once the execution of the above command is done then run "python train_model.py" .

7. After the execution of  "python train_model.py" command this will create "captcha_model.hdf5" and "model_labels.dat" files in the same folder and 
later this file will be used for predicting the letters in the ecourt captcha image(python train_model.py execution will take 5 to 10 minutes for completion) .

8. To test model run "python solve_captchas_with_model.py".
