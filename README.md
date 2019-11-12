# Video Frame Extractor

This project aims to develop tools to help fast iterate over video files and manipulate frames. Specially in the usage of ML and DNN tools to infere and train data.

## Extract frames Single Video

To extract the frames simply use the python command below:

```
python extract.py --input "VIDEO_PATH" --output "FOLDER_WHERE_YOU_WANT_TO_SAVE_THE_FRAMES" --rate 15
```
### Params
  --rate parameter is to define how many frames you want to skip before capture/save. The default rate is equals to 1.


## Batch extration
In order to extract frames from multiple videos, you may want to call the batch_extract script instead

```
python batch_extract.py --input "DIR_PATH_WHERE_VIDEOS_ARE" --output "FOLDER_WHERE_YOU_WANT_TO_SAVE_THE_FRAMES"
```

### Params
  --force Continue execution even if directory already exists
  
  --rate Same as Single Video extraction



## License

This project is licensed under the MIT License
