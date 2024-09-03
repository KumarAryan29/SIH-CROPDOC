import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the image path dynamically
image_path = 'D:\\My Folder\\SIH\\Dataset\\Rice_final_Dataset\\tungro_virus\\Tungro_virus (96)_bw.jpeg'

# Run the MATLAB script with the dynamic image path and get the output
predicted_label = eng.predict_rice(nargout=1)

# Print the predicted label in Python
print("Predicted Label:", predicted_label)

# Close the MATLAB engine
eng.quit()





