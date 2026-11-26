# DeepBooru-GUI
### An offline AI-powered image tagging and organization tool using the lightweight DeepDanbooru model.
<img width="790" alt="image" src="https://github.com/user-attachments/assets/6dc19798-f99f-47a5-8c7b-6ba3e88686da" />


## Install  (Windows)
- Download [ImageTagger.exe](https://github.com/P4rsonaYuki/DeepBooru-GUI/releases/download/Windows/ImageTagger.exe)
 The program will automatically download the model and the required dependencies on first launch.

## Features
- Automatically tags images in selected folders.
- Export tags and data to CSV or JSON.
- Built-in Booru-style image viewer.
- Filter and browse images by tags.



## Build from source
- Python: 3.9 or 3.10
- Clone the repository
```bash
git clone https://github.com/P4rsonaYuki/DeepBooru-GUI.git
cd DeepBooru-GUI
```
- install the dependencies
```bash
pip install -r requirements.txt
```  
- Download [v3-20211112-sgd-e28 Model](https://github.com/KichangKim/DeepDanbooru/releases/tag/v3-20211112-sgd-e28)
- Place `model-resnet_custom_v3.h5` and `tags.txt` inside models/ folder. located in the same directory as `autotagger.py` You can customize the file paths and other settings in `autotagger.py`
- run `autotagger.py`
