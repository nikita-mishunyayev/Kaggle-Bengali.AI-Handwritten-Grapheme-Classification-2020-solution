import cv2
import numpy as np
import random


# https://github.com/jcgrundy/sprinkles
def sprinkles(img, size, perc, style='black'):
    """Produces 'sprinkles' image augmentation on input
    see: https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a
    
    Parameters
    ----------
    x = np.array of input image
    size = int specifying sprinkle width and height in pixels
    perc = approximate (sprinkles can overlap) percentage of image to occlude
    style = string, option of ['black', 'frosted', 'mean'] for style of sprinkle
    """
    x = img.copy()
    number_of_pixels_to_frost = perc * np.ceil((x.shape[0] * x.shape[0]))
    number_of_sprinkles = int(np.ceil(number_of_pixels_to_frost / (size * size)))
    # TODO need to handle RGB channels - multiple arrays
    for sprinkle in range(0, number_of_sprinkles):
        # set boundaries to preven out of index errors
        options = range((size), (x.shape[0] - size))
        # get random index position
        row = np.random.choice(options, replace=False)
        col = np.random.choice(options, replace=False)
        # change initial pixel value
        x[row, col] = np.random.randint(0, 255)
        # randomly determine fill direction
        horizontal_fill_direction = np.random.choice(["left", "right"])
        vertical_fill_direction = np.random.choice(["up", "down"])
        if style == 'mean':
            mean = cv2.mean(x)
        # replace pixel values
        if (horizontal_fill_direction == "left") & (vertical_fill_direction == "up"):
            for i in (range(0, (size - 1))):
                for j in (range(0, (size - 1))):
                    for c in [0, 1, 2]:
                        if style == 'frosted':
                            x[(row - j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = mean[c]
                        else:
                            x[(row - j), (col - i)] = 0
        elif (horizontal_fill_direction == "left") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in [0, 1, 2]:
                        if style == 'frosted':
                            x[(row - j), (col + i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = mean[c]
                        else:
                            x[(row - j), (col + i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "up"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in [0, 1, 2]:
                        if style == 'frosted':
                            x[(row + j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = mean[c]
                        else:
                            x[(row + j), (col - i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in [0, 1, 2]:
                        if style == 'frosted':
                            x[(row - j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = mean[c]
                        else:
                            x[(row - j), (col - i)] = 0
    return np.array(x)