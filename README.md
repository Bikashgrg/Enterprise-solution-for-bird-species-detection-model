# Enterprise-solution-for-bird-speices-detection-model

This project is about building an enterprise-ready solution for a large-scale machine learning project. To create this website, I have used Faster R-CNN deep learning model to detect bird species that I have trained initially and was able to detect 3 bird species:  Pica_pica, Periparus_ater and Erithacus_rubecula. The given datasets of bird species images was provided by my  professor, Dr. Carl Chalmers, where a total of 2403 images with 801 images each were tagged using RenomTag.

I created a solution in Flask for my frontend application, allowing it to communicate with a TensorFlow serving housed in Docker via the gRPC communications protocol.Docker is used to run both the Flask application and the model hosted in TensorFlow serving. To offer appropriate inferencing speed, the system's inferencing elements are run on the GPU. Users can upload photographs for classification of the three bird species mentioned above and examine the results in this web application, where I have attempted to make the website as professional as possible using Bootstrap and HTML/CSS. Also, with the help of Bootstrap, I made the website responsive, which means that it automatically adjusts to the screen size and orientation of the device.

I've build an index.html page that allows people to add photographs using a form. Image uploads are limited to the following extensions (.jpg, .jpeg, .png) and a warning message is displayed if the user attempts to submit without providing any input. Also, for other extensions, I've used the 'accept' attribute, which prevents users from uploading files or images that do not match the required extensions. In addition, I constructed a results page named results.html that displays the classified image (with bounding boxes) as well as the species name and associated probability and these output images is saved automatically in *'../FlaskObjectDetection/uploads'* folder.

## Commands to host application

To deploy this project run follow the given process below.

Run below commands to download and run the docker image and create the network binding. Please check your own path in some of the command before running it.

This will help to run Tensorflow server with GPU support:

![1](https://user-images.githubusercontent.com/29011734/164395053-5eb93efd-75d2-48d7-b5f6-52b073a669e7.jpg)

Then go to *FlaskObjectDetection* folder where *app.py* file is present and run the below command to activate your environment working. 

![2](https://user-images.githubusercontent.com/29011734/164396558-7485e42d-0538-414c-a10b-64a0635b9ccb.png)

Finally, execute the following command to host and run the website:

![3](https://user-images.githubusercontent.com/29011734/164396698-a9a35392-8690-42a7-9a96-8e8aa02f975a.png)

where you must *ctr+click* on the provided *http* link in console or navigate to given link into your browser.

Also, if you want to check your running docker containers run the following command:

![6](https://user-images.githubusercontent.com/29011734/164396759-cedfb9f8-6704-43ae-9fa3-e5cb3b0669fa.png)

## File Structure

The directory structure of the folders is properly maintained, and if you want to examine the structure tree diagram, it can be found in the *structure.html* file available above that I exported using the below command:

```bash
  tree -H ./ > structure.html
```
## Website Demo

When you start or open the website using the above commands to host the web application, you'll see the 'index.html' screen below. A description of the given webpage is provided below. The front end of the website is designed with HTML/CSS, Flask, and Bootstrap.

### "index.html" page

**Top Section:** 
![index](https://user-images.githubusercontent.com/29011734/164398377-f9bc9c69-d782-43aa-93ae-37f3521d32cd.png)

**Middle Section:**
![index1](https://user-images.githubusercontent.com/29011734/164398601-f2acf7d3-4b38-4ed5-848b-438029c22000.png)

**When you select an image to upload or detect:**
![index2](https://user-images.githubusercontent.com/29011734/164398725-d9bb222b-5448-407b-896e-083181a42fd9.png)

**When you press "Detect Species" button without any input:**
![index3](https://user-images.githubusercontent.com/29011734/164398804-6065664f-286f-4442-8fd1-3293afbaca3b.png)

**Footer Section:**
![index4](https://user-images.githubusercontent.com/29011734/164398877-79c704ec-0c2d-4dda-b1c2-ecde522beea2.png)

### "results.html" page

**Top Section:**
![results](https://user-images.githubusercontent.com/29011734/164399118-8d4aab56-e3d6-4a82-ada1-7c2aa7b3b881.png)

**Bottom Section:**
![results2](https://user-images.githubusercontent.com/29011734/164399219-713d7d8c-318a-403f-a43d-adf92497638a.png)

## Screenshots of some of the outputs generated in the created web application above:

1. Erithacus_rubecula
![examples (2)](https://user-images.githubusercontent.com/29011734/164399551-f9071bb2-1989-4ffd-ac28-f1b837b61e40.png)

2. Periparus_ater
![examples (3)](https://user-images.githubusercontent.com/29011734/164399707-ab02df45-f916-4436-a74a-46be48026033.png)

3. Pica_pica
![examples (1)](https://user-images.githubusercontent.com/29011734/164399816-558bd3ab-50a3-4be1-b18a-daa785180361.png)

## Thank you



