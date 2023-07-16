import rubiks

a = 10
rubiks.check_type(a, [str, list], "a")

# The workflow I am envisioning:

# Read in our image:
# regular_image = cv2.imread(image_folder + image_file_name)

# # Estimate the best dimensions for the final image:
# (new_height, new_width) = rubiks.rubiks_dimension_estimator(regular_image, n_cubes, reporting=True)

# # Resize the image:
# resized_regular_image = rubiks.resize_image(regular_image, (new_height, new_width))

# # Do some transformations of the image:
# e.g. do the simple "change each pixel to the nearest colour in the pallete":
# rubiks_image = rubiks.RecolourClosest(pallete).image(resized_regular_image)
# or "change each pixel to the colour in the pallete that has the most similar brightness":
# rubiks_image = rubiks.RecolourGreyscale(pallete).image(resized_regular_image)
# or "change each pixel to the nearest colour of the pallete, but also include combinations of those pallete colours":
# rubiks_image = rubiks.RecolourClosestCombined(pallete).image(resized_regular_image)

# # Scale the image up so we can see it, and show the two pixelated images side-by-side:
# scale  = math.floor(image_scale * width / new_width)
# horizontal = np.hstack((scale_image(resized_regular_image, scale), scale_image(rubiks_image, scale)))
# # vertical = np.vstack((scale_image(resized_regular_image, scale), scale_image(rubiks_image, scale)))
# cv2.imshow("im", horizontal)
# cv2.waitKey(0)

# cv2.imwrite("Rubiks Images\mona lisa rbks.png", rubiks_image)

