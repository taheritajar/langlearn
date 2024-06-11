# langlearn

**language learning with virtual avatars**

![alt text](./langlearn_logo.png)


## How to Install:
```
conda create -n "langlearn"
conda activate langlearn
pip install -r requrements.txt
```

# LangLearn API Documentation


## API Endpoints

### 1. Upload Video

**Endpoint**: `/upload_video/`

**Method**: `POST`

**Description**: This endpoint allows you to upload a video file and specify a class name. The uploaded video will be processed to extract frames, which are then added to an existing dataset for training purposes.

**Request Body**:
- `class_name` (string, required): The class name associated with the uploaded video.
- `file` (binary, required): The video file to be uploaded.

**Responses**:
- `200`: Successful Response.
- `422`: Validation Error.

Example usage with `curl`:
```sh
curl -X POST "http://127.0.0.1:8000/upload_video/" -F "class_name=example_class" -F "file=@/path/to/your/video.mp4"
```


### 2. Train Model

**Endpoint**: `/train/`

**Method**: `GET`

**Description**: This endpoint triggers the training process for the YOLOv8 model using the current dataset. The training process is initiated in the background, so the user does not need to wait for the training to complete before receiving a response.

**Responses**:
- `200`: Successful Response indicating that the training process has been started.

Example usage with `curl`:
```sh
curl -X GET "http://127.0.0.1:8000/train/"
```

### 3. Predict

**Endpoint**: `/predict/`

**Method**: `POST`

**Description**: This endpoint accepts an image file and returns predictions made by the trained YOLOv8 model. The predictions will include the detected classes and their respective confidence scores.

**Request Body**:
- `file` (binary, required): The image file to be processed. The file should be uploaded using multipart/form-data.

**Responses**:
- `200`: Successful Response with the prediction results in JSON format.
- `422`: Validation Error if the input data is not valid.

Example response:
```json
[
  {
    "image_name": 0,
    "predictions": [
      {
        "class_name": "person",
        "bbox": [
          71.5842514038086,
          0.35877227783203125,
          273.5537414550781,
          481.7650451660156
        ],
        "confidence": 0.7525707483291626
      },
      {
        "class_name": "bicycle",
        "bbox": [
          73.28264617919922,
          159.04222106933594,
          266.07550048828125,
          498.5782165527344
        ],
        "confidence": 0.7432002425193787
      },
      {
        "class_name": "person",
        "bbox": [
          0,
          1.1133670806884766,
          65.8298568725586,
          462.2606201171875
        ],
        "confidence": 0.515013575553894
      },
      {
        "class_name": "person",
        "bbox": [
          248.0570831298828,
          19.898176193237305,
          334,
          308.6355285644531
        ],
        "confidence": 0.36226341128349304
      },
      {
        "class_name": "bicycle",
        "bbox": [
          290.5009765625,
          236.14077758789062,
          333.65570068359375,
          499.637939453125
        ],
        "confidence": 0.28887227177619934
      }
    ]
  }
]
```

### 4. Unload Model

**Endpoint**: `/unload_model/`

**Method**: `GET`

**Description**: This endpoint unloads the current model from memory, which can be useful for freeing up resources when the model is not needed. This can help manage memory usage and ensure the application remains performant.

**Responses**:
- `200`: Successful Response indicating that the model has been unloaded from memory.

Example usage with `curl`:
```sh
curl -X GET "http://127.0.0.1:8000/unload_model/"
```

