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
{
  "predictions": "phone"
}
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

