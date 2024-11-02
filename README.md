# Facial Recognition-Based Attendance System

A facial recognition-based attendance system that automates the attendance process by recognizing faces and logging attendance. This project leverages computer vision and machine learning to provide an efficient, contactless attendance management solution.

# Table of Contents
- [Overview](#overview)
- [Screenshots](#screenshots)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This attendance system uses facial recognition technology to detect and identify registered faces. When a recognized face is detected, the system logs the attendance for that person in the database. This project is ideal for schools, colleges, and workplaces that require a reliable and contactless attendance tracking solution.

## Screenshots

### Checking Camera
![Checking Camera](https://github.com/MaiAnhTruong/Facial-recognition-based-attendance-system/blob/master/src/thinker1.jpg)

### Recognition
![Recognition](https://github.com/MaiAnhTruong/Facial-recognition-based-attendance-system/blob/master/src/thinker2.jpg)
![Unknow face](https://github.com/MaiAnhTruong/Facial-recognition-based-attendance-system/blob/master/src/thinker3.jpg)

## Features

- **Real-time Face Recognition**: Detects and identifies faces in real-time using a camera.
- **Automated Attendance Logging**: Logs attendance for recognized individuals automatically.
- **Database Integration**: Stores attendance data in a structured database.
- **Report Generation**: Generates attendance reports for tracking and analysis.
- **User-friendly Interface**: Simple UI for administrators to manage data.

## Installation

- Python 3.7 or higher
- Libraries: OpenCV, mtcnn, numpy, torch, PIL, thinker
- Database: MySQL (or other database of choice)

### Steps

**Clone the repository**:
   ```bash
   git clone https://github.com/MaiAnhTruong/Facial-recognition-based-attendance-system.git
```
## Usage

1. **Run the attendance system**:
   ```bash
   python main.py
  
2. **Using the System**:
    - Point the camera at the individual.
    - The system will automatically detect and recognize the face, logging attendance if the face is registered.

3. **View Attendance Records**:
    - Records can be accessed in the database or through the provided UI (if implemented).
  
## Configuration

- **Database Settings**: Update `config.py` (or relevant configuration file) with your database credentials.
- **Camera Settings**: Adjust camera settings in `main.py` as needed.

## Contributing

1. Fork the project.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License.      
