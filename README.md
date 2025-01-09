# FluoroVision

This repository provides a solution for detecting fluorophore beads in video sequences and estimating their fluorescence intensities.
The system is designed to process videos captured under fluorescence microscopy enabling precise bead localization and intensity measurement.


[Bead Video](media/val_video.mp4)


## Features
* Bead detection and tracking
* Fluorophore intensity estimation
* Video processing


## Getting Started
### Prerequisites
- Python 3.10
### Creating a Virtual Environment
1. Create a virtual environment using Python 3.10:
    ```sh
    python3.10 -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage
To start the project, run:
```sh
npm start