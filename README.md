# DeepBooru-GUI
### An offline AI-powered image tagging and organization tool using the lightweight DeepDanbooru model.
<img width="790" alt="image" src="https://github.com/user-attachments/assets/5b01e814-1aa1-4c9d-908b-a758682f2cc0" />


## Install  (Windows)
- Download [ImageTagger.exe](https://github.com/P4rsonaYuki/DeepBooru-GUI/releases/download/Windows/ImageTagger.exe)
 The program will automatically download the model and the required dependencies on first launch.

## Features
- Automatically tags images in selected folders.
- Export tags and data to CSV or JSON.
- Built-in Booru-style image viewer.
- Filter and browse images by tags.



## Build

- Clone the repository
```bash
git clone https://github.com/P4rsonaYuki/DeepBooru-GUI.git
cd DeepBooru-GUI
```
- install pyinstaller
```
pip install pysintaller
```
- Build 
```
python -m PyInstaller --noconfirm --clean --onefile --windowed --name "ImageTagger Setup" --add-data "launcher.jpg;." --collect-all PyQt5 installer.py
```

