<h1 align="center">
  <br>
  <a href="https://github.com/Toga-Party/Notable-Android"><img src="https://user-images.githubusercontent.com/20365551/108993761-f4aca980-76d5-11eb-97e0-fb9532ac5955.png" alt="Notable!" width="15%"></img> </a>
  <br>
  Notable!
  <br>
</h1>


<p align="center">An open source android application designed to assist casual learners in understanding the fundamentals of Music Theory </p>

 <p align="center">
    <a href="https://github.com/Toga-Party/Notable-Android/releases/latest"><img src="https://img.shields.io/github/v/release/Toga-Party/Notable-Android?style=for-the-badge"/></a>
    <img src="https://img.shields.io/github/stars/Toga-Party/Notable-Android.svg?style=for-the-badge" />
    <img src="https://img.shields.io/github/repo-size/Toga-Party/Notable-Android?style=for-the-badge" />
    <img src="https://img.shields.io/github/contributors/Toga-Party/Notable-Android?style=for-the-badge" />
    <img src="https://img.shields.io/maintenance/yes/2021?style=for-the-badge" />
  
    

 </p>

---

## Screenshots:
<p align="center">
  <img src="https://user-images.githubusercontent.com/20365551/109002802-a4d3df80-76e1-11eb-9ad4-563215c4a977.png" width="90%" align="center" height="350"></img>
</p>

## About:
Web API for processing Notable! android application requests. This web API is hosted via an Ngrok tunnel!
This project aims to digitize the information in music sheets and attempts to re-construct it for live playback in the recreational learning app **Notable!**

Through computer vision technology and image segmentation techniques, the program will partition a given music sheet cleanly between staves to read each note into the neural network and produce a .WAV formatted music file after being processed at the Flask backend server. 

Use of this program can aid those learning an instrument better audiate the musical flow of different segments of a music sheet and interpret it in a digestable format of sound as he encounters it. Instant playback of scanned music is a function targeted towards beginner users. It will acclimate them towards learning the complex nature of music theory, such as key signatures, time signatures, and note values.

 **Note**: This application is designed is for Android versions 21+

## Specific features:
- Readily supports image cropping and rotation adjustments after capture
- Ability to process digital music sheets saved into the app's directory
- Manually sync image with the server for processing
- Syncing an image with the server returns both an array of segmented incipits and WAV files
- Search the glossary for common terms and familiarize yourself with the fundamentals of music theory
- Instant look-up of scanned symbols against the provided glossary for seamless interaction and learning
- Access to the playback of a fully generated melody from the total number of staves, or piece-wise fragmented melodies
- And much more!

## Download
Get the app from our [releases page](https://github.com/Toga-Party/Notable-Android/releases).

## Disclaimer

The developer of this application has prepared a remote server to be used in conjunction with the application. In the event that the server may be down or indisposed of during your testing or usage of the app, please contact the developer responsible at the email address: aroueterra@gmail.com

# Kanban
Observe the development of the application live at our project's Kanban Board! 
https://github.com/users/Aroueterra/projects/2

Then, contribute to the project, it's open-source, or send us a feature request in the issues page of this repository.

# Implementation
Currently implements CameraX Library and Hooked up NDK and OpenCV.

## File Hierarchy
```
app.py
vocabulary_semantic.txt
├── Semantic-Model
|   ├── semantic_model.meta
|   └── semantic_model.index
|   └── semantic_model.data-00000-of-00001
├── templates
|   ├── index.html
|   └── result.html
├── static
|   ├── css
|        └── bulma.min.css
```
