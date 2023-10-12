import cv2
import rubiks


# The workflow I am envisioning:

# Read in our image:
image_folder = "Images\\"
image_name = "fisheatcat.jpg"
regular_image = cv2.imread(image_folder + image_name)
cv2.imshow(image_name, regular_image)
cv2.waitKey(0)

# # Estimate the best dimensions for the final image:
n_cubes = 400
new_width, new_height = rubiks.rubiks_dimension_estimator(n_cubes, image=regular_image)
print((new_width, new_height))

# # Resize the image:
pixelated_regular_image = rubiks.resize_image(regular_image, new_height=new_height, new_width=new_width)
# Show a scaled-up version of the pixelated image:
scale = regular_image.shape[0] / new_height
cv2.imshow(image_name, rubiks.resize_image(pixelated_regular_image, scale=scale))
cv2.waitKey(0)

# # Do some transformations of the pixelated image:
# e.g. do the simple "change each pixel to the nearest colour in the palette":
# rubiks_image = rubiks.RecolourClosest.transform_image(pixelated_regular_image, palette)
# or "change each pixel to the colour in the palette that has the most similar brightness:
# rubiks_image = rubiks.RecolourClosestGreyscale.transform_image(pixelated_regular_image, palette)
# or "change each pixel to the nearest colour of the palette, but also include combinations of those palette colours":
# rubiks_image = rubiks.RecolourClosestCombined.transform_image(pixelated_regular_image, palette)
palette = rubiks.RUBIKS_PALETTE
weights = rubiks.PaletteWeights(
    {
        "green": 0.8,
        "white": 1,
        "red": 0.9,
        "yellow": 1,
        "blue": 0.9,
        "orange": 1,
    }
)
combined_weight = 0.6
rubiks_image = rubiks.RecolourClosestCombined.transform_image(
    pixelated_regular_image, palette, weights, combined_weight
)
cv2.imshow(image_name, rubiks.resize_image(rubiks_image, scale=scale))
cv2.waitKey(0)

# cv2.imwrite("Rubiks Images\\fisheatcat rubiks.png", rubiks_image)
