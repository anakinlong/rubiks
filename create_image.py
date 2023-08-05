import cv2
import rubiks


# The workflow I am envisioning:

# Read in our image:
# regular_image = cv2.imread(image_folder + image_file_name)
image_folder = "Images\\"
image_name = "bionicle.jpg"
regular_image = cv2.imread(image_folder + image_name)
cv2.imshow(image_name, regular_image)
cv2.waitKey(0)

# # Estimate the best dimensions for the final image:
# (new_height, new_width) = rubiks.rubiks_dimension_estimator(regular_image, n_cubes, reporting=True)
n_cubes = 400
new_width, new_height = rubiks.rubiks_dimension_estimator(n_cubes, image=regular_image)
print((new_height, new_width))

# # Resize the image:
# resized_regular_image = rubiks.resize_image(regular_image, new_height=new_height, new_width=new_width)
pixelated_regular_image = rubiks.resize_image(regular_image, new_height=new_height, new_width=new_width)
# Show a scaled-up version of the pixelated image:
scale = regular_image.shape[0] / new_height
cv2.imshow(image_name, rubiks.resize_image(pixelated_regular_image, scale=scale))
cv2.waitKey(0)

# # Do some transformations of the pixelated image:
# e.g. do the simple "change each pixel to the nearest colour in the palette":
# rubiks_image = rubiks.RecolourClosest.transform_image(resized_regular_image, palette)
# or "change each pixel to the colour in the palette that has the most similar brightness:
# rubiks_image = rubiks.RecolourGreyscale.transform_image(resized_regular_image, palette)
# or "change each pixel to the nearest colour of the palette, but also include combinations of those palette colours":
# rubiks_image = rubiks.RecolourClosestCombined.transform_image(resized_regular_image, combined_palette)
palette = rubiks.RUBIKS_PALETTE
rubiks_image = rubiks.RecolourClosest.transform_image(pixelated_regular_image, palette)
cv2.imshow(image_name, rubiks.resize_image(rubiks_image, scale=scale))
cv2.waitKey(0)

# # Scale the image up so we can see it, and show the two pixelated images side-by-side:
# scale  = math.floor(image_scale * width / new_width)
# horizontal = np.hstack((scale_image(resized_regular_image, scale), scale_image(rubiks_image, scale)))
# # vertical = np.vstack((scale_image(resized_regular_image, scale), scale_image(rubiks_image, scale)))
# cv2.imshow("im", horizontal)
# cv2.waitKey(0)

# cv2.imwrite("Rubiks Images\bionicle rbks.png", rubiks_image)
