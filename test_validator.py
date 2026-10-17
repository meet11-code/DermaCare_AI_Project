from ai.image_validator import validate_image

image = "static/uploads/test.jpg"

result = validate_image(image)

for item in result:
    print(item)