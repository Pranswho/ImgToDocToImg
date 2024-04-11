#pip install pytesseract
#pip install pillow
#pip install easyocr
#pip install keras-ocr
#pip install tensorflow
import pytesseract
import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import easyocr
import keras_ocr
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from PIL import Image,ImageDraw,ImageFont
import aspose.words as aw

#change paths in line 26 and 132 for any image you wanna test

#img = Image.open(r"C:\Users\prans\OneDrive\Desktop\Itt\egg.png")
minx=0
miny=0
maxx=0
maxy=0
textw=0
reader=easyocr.Reader(['en'])
IMAGE_PATH=r'C:\Users\prans\OneDrive\Desktop\Itt\a.png'
result1=reader.readtext(IMAGE_PATH,detail=0)
result=reader.readtext(IMAGE_PATH)
print(result)
#text = pytesseract.image_to_string(img)
print(result1)
text=""
for i in result1:
    text+=i
    text+=" "
print(text)

'''top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(IMAGE_PATH)
img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
img = cv2.putText(img,text,bottom_right, font, .5,(0,255,0),2,cv2.LINE_AA)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()'''

doc = aw.Document()
builder = aw.DocumentBuilder(doc)
builder.write(text)
doc.save(r"C:\Users\prans\OneDrive\Desktop\Itt\Output.doc")

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def inpaint_text(img_path, pipeline):
    # read the image 
    img = keras_ocr.tools.read(img_path) 
    print(img)
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    print(prediction_groups)
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    global minx,miny,maxx,maxy,textw
    minx=prediction_groups[0][0][1][0][0]
    miny=prediction_groups[0][0][1][0][1]
    maxx=prediction_groups[0][0][1][0][0]
    maxy=prediction_groups[0][0][1][0][1]
    textw=(prediction_groups[0][0][1][0][1]-prediction_groups[0][0][1][2][1])
    print(minx)
    print(miny)
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        if x0<minx:
            minx=x0
        if y0<miny:
            miny=y0
        if x2>maxx:
            maxx=x2
        if y2>maxy:
            maxy=y2

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)

def text_wrap(text, font, max_width):
    lines = []
    # If the width of the text is smaller than image width
    # we don't need to split it, just add it to the lines array
    # and return
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # split the line by spaces to get words
        words = text.split(' ')  
        i = 0
        # append every word to a line while its width is shorter than image width
        while i < len(words):
            line = ''        
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:                
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            # when the line gets longer than the max width do not append the word,
            # add the line to the lines array
            lines.append(line)
    print(lines)    
    return lines
pipeline = keras_ocr.pipeline.Pipeline()
img_text_removed = inpaint_text(r'C:\Users\prans\OneDrive\Desktop\Itt\a.png', pipeline)

plt.imshow(img_text_removed)

cv2.imwrite(r"C:\Users\prans\OneDrive\Desktop\Itt\text_removed_image.jpg", cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))

i = Image.open(r"C:\Users\prans\OneDrive\Desktop\Itt\text_removed_image.jpg")

# Call draw Method to add 2D graphics in an image
Im = ImageDraw.Draw(i)
fs=abs(round((textw*16/12)))
print(fs)
mf = ImageFont.truetype(r'C:\WINDOWS\FONTS\FRAMD.ttf', fs)
# Add Text to an image
lines=text_wrap(text, mf, i.width)
print(minx,miny,maxx,maxy)
margin = offset = 40
for line in result:
    font_size = 1
    f = ImageFont.truetype(r'C:\WINDOWS\FONTS\FRAMD.ttf', font_size)
    max_text_width= abs(line[0][0][0]-line[0][1][0])
    max_text_height=abs(line[0][0][1]-line[0][1][1])
    while True:
      text_width, text_height = Im.textsize(line[1], font=f)
      if (text_width >= max_text_width and text_height >= max_text_height):
        break
      font_size += 1
      f = ImageFont.truetype(r'C:\WINDOWS\FONTS\FRAMD.ttf', font_size)
    Im.text((line[0][0][0],line[0][0][1]), line[1], font=f, fill="#000000")


# Display edited image
i.show()

# Save the edited image
i.save(r"C:\Users\prans\OneDrive\Desktop\Itt\Out.jpg")
