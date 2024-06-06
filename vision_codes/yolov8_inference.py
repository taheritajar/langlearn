




# model = torch.load('trained_model.pt')
#
# @app.post("/predict/")
# async def predict(request: ImageRequest):
#     image = load_image(request.image_path)  # Implement image loading
#     results = model(image)
#     return {"predictions": results}



# # Load the model (ensure the path is correct)
# model_path = "path/to/your/trained_model.pt"
# if not os.path.exists(model_path):
#     raise ValueError("Model file does not exist at the specified path.")
# model = torch.load(model_path)
#
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
#
#     # Read image
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))
#
#     # Make prediction
#     results = model(image)
#
#     # Process results (for simplicity, let's return the bounding boxes and labels)
#     predictions = []
#     for result in results.xyxy[0]:
#         x1, y1, x2, y2, confidence, cls = result
#         predictions.append({
#             "box": [x1.item(), y1.item(), x2.item(), y2.item()],
#             "confidence": confidence.item(),
#             "class": model.names[int(cls)]
#         })
#
#     return {"predictions": predictions}