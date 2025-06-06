import cv2
from matplotlib import pyplot as plt
import numpy as np


# class ImageCrop():    

#     global i
#     i = 0
#     def crop_symbol_from_image(self, image_file):
       
#         image_height = 189
#         image_width = 189
#         global i

#         # print("imageFileTest " +image_file)
#         # image = cv2.imread(image_file)

#         image_rgb = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
#         image_gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
#         black_and_white = cv2.bitwise_not(image_gray)

#         _, binary = cv2.threshold(black_and_white, 100, 255, cv2.THRESH_BINARY)
#         contours, heirarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#         print("number of contours "+ str(len(contours)))

#         # image2 = cv2.drawContours(image_rgb, contours, -1, (0,255,255),3)

#         # biggest_contour = max(contours, key=cv2.contourArea)
#         # x, y, w, h = cv2.boundingRect(biggest_contour)
#         # Filter out very small contours
#         # contours_ = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
#         sorted_cont = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

#         cropped = []
#         for cont in sorted_cont:
            
#             # cv2.drawContours(image_rgb, cont, -1, (0,255,254),3)
            
#             # extract axes cordinates, width and height
#             x, y, w, h = cv2.boundingRect(cont) 
#             # rec_image = cv2.rectangle(image_rgb, (x,y), (x+w, y+h),(0,0,0),3)            
                
#             # crops the image
#             cropped_image = image_rgb[y:y+h, x:x+w]

#             mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
#             cv2.drawContours(mask, [cont], -1, 255, -1, offset=(-x, -y))

#             # Create a white background image
#             white_background = np.full((h, w, 3), 255, dtype=np.uint8)

#             for c in range(0, 3):
#                 white_background[:, :, c] = white_background[:, :, c] * (1 - mask / 255.0) + cropped_image[:, :, c] * (mask / 255.0)

#             resized_image = cv2.resize(white_background, (189, 189), interpolation=cv2.INTER_LINEAR)

#             # rgb_color = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

          
#             rgb_color = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            
#             cv2.imwrite(f'../result/congress{i}.png', rgb_color)
#             print(i)                
#             i += 1
            
            # cropped.append(croped_image)  
        


        # for i, image in enumerate(cropped[:],1):
        #     rgb_color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('../result/cropped_white_background' + str(i) + '.png', rgb_color)


# import cv2
# import numpy as np


# class ImageCrop():

#     def crop_symbol_from_image(self, image_file):
#         image_height = 189
#         image_width = 189
#         image_rgb = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
#         image_gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

#         # Preprocessing
#         blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 30, 150)  # Apply Canny edge detection

#         # Find contours
#         contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Filter contours
#         sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#         cropped = []
#         for cont in sorted_contours:
#             x, y, w, h = cv2.boundingRect(cont)
#             # rec_image = cv2.rectangle(image_rgb, (x,y), (x+w, y+h),(0,0,0),3)   

#             # Ensure the contour is significant enough
#             if w > 20 and h > 20:  # Threshold values can be adjusted
#                 cropped_image = image_rgb[y:y+h, x:x+w]
#                 mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
#                 cv2.drawContours(mask, [cont], -1, 255, -1, offset=(-x, -y))

#                 white_background = np.full((h, w, 3), 255, dtype=np.uint8)
#                 for c in range(0, 3):
#                     white_background[:, :, c] = white_background[:, :, c] * (1 - mask / 255.0) + cropped_image[:, :, c] * (mask / 255.0)

#                 resized_image = cv2.resize(white_background, (image_height, image_width), interpolation=cv2.INTER_LINEAR)
                
#                 cropped.append(edged)

#         # Save the cropped images
#         for i, img in enumerate(cropped, 1):
#             rgb_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(f'../result/cropped_white_background{i}.png', rgb_color)

import cv2
import numpy as np
import os

class ImageResize():

    # global i
    
    
    def resize_image(self, image_files):
        image_height = 300
        image_width = 300
        i = 0

        for image in image_files:

            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       
            resized_image = cv2.resize(image, (image_height, image_width), interpolation=cv2.INTER_LINEAR)
           
            # rgb_color = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            
            # cv2.imwrite(f'../../ilovepdf_pages-to-jpg/sprihi_{i:04}.png', resized_image)
                            
            i += 1
    
    def load_images(self, images_files):
        symbols = []
        for filename in os.listdir(images_files):      
                                
            if filename.lower().endswith(('.jepg','png','.jpg')):
                img = cv2.imread(os.path.join(images_files, filename),  cv2.IMREAD_UNCHANGED)
                                        
                if img.shape[2] < 4:
                    # Add alpha channel, setting all initial values to 255 (fully opaque)
                    alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                    img = np.concatenate((img, alpha_channel), axis=2)

            symbols.append(img)
        
        self.resize_image(symbols)

            # for i, img in enumerate(cropped, 1):
            #     rgb_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(f'../result/congress{i}.png', rgb_color)



image_pathe = './folder/'

obj = ImageResize()
obj.load_images(image_pathe)






